import os
import numpy as np
import time
import argparse
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import models.resnet
import models.vit_dino
import models.vit
from utils.YParams import YParams
from utils.data_loader_rgb import get_data_loader

from torch.optim import lr_scheduler
from utils.scheduler import GradualWarmupScheduler
from astropy.stats import mad_std

def set_seed(params):
  seed = params.seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if params['world_size'] > 0:
      torch.cuda.manual_seed_all(seed)

def count_parameters(model):
  params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return params/1000000

class Trainer():

  def __init__(self, params):
    self.params = params
    self.device = torch.cuda.current_device()

    # first constrcut the dataloader on rank0 in case the data is not downloaded
    logging.info('rank %d, begin data loader init'%params.world_rank)
    self.train_data_loader, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), load_specz=True, is_train=True)
    self.valid_data_loader, self.valid_sampler = get_data_loader(params, params.valid_data_path, dist.is_initialized(), load_specz=True, is_train=False)
    logging.info('rank %d, data loader initialized'%params.world_rank)

    if params.model == 'resnet':
      self.model = models.resnet.resnet50(num_channels=params.num_channels, num_classes=params.num_classes).to(self.device)
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    elif params.model == 'vit':
      self.model = models.vit.ViT(image_size=params.crop_size, num_classes=params.num_classes, channels=params.num_channels,
        patch_size=params.patch_size,
        dim=params.embed_dim,   
        depth=params.depth,
        heads=params.num_heads,
        mlp_dim=params.mlp_dim,
        dropout=0.1,
        emb_dropout=0.1).to(self.device)
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
      #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    elif params.model == 'vit_dino':
      if not os.path.exists(params.pretrained_model_path):
        logging.warning("pretrained model does not exist!")
        exit(1)
      else:
        pretrained_model_path = params.pretrained_model_path
        logging.info("loading pretrained model at {}".format(pretrained_model_path)) 
        self.model = models.vit_dino.vit_small_pretrained(img_size=[224], in_chans=params.num_channels,
                    num_classes=params.num_classes,
                    patch_size=params.patch_size,
                    drop_path_rate=params.stoch_drop_rate,
                    drop_rate=0.1,
                    attn_drop_rate=0.1,
                    pretrained_model_path=pretrained_model_path,
                    ).to(self.device)
      parameters = []
      feature_extractor_parameters = []
      for name, param in self.model.named_parameters():
        if "head" in name:
          # the finetuned head
          parameters.append(param)
        else:
          # the backbone
          feature_extractor_parameters.append(param)

      # the backbone learns at a lower learning rate
      feat_lr = params.lr * 0.1
      self.optimizer = torch.optim.SGD([{'params': parameters},
                                        {'params': feature_extractor_parameters, 'lr': feat_lr}],
                                       lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    else:
      logging.warning("model architecture invalid")
      exit(1)
      

    if params.model == 'vit' or params.model == 'vit_dino':
      # cosine scheuler from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch 
      scheduler_cosine = lr_scheduler.CosineAnnealingLR(self.optimizer, self.params.max_epochs - 1)
      self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
    else:
      self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=params.lr_milestones, gamma=0.1)

    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    if params.amp:
      self.grad_scaler = torch.cuda.amp.GradScaler()

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank],
                                           output_device=[params.local_rank])
    self.iters = 0
    self.startEpoch = 0
    if params.resuming:
      logging.info("Loading checkpoint %s"%params.checkpoint_path)
      self.restore_checkpoint(params.checkpoint_path)
    self.epoch = self.startEpoch

    if params.log_to_screen:
      logging.info(self.model)

    if params.log_to_tensorboard:
      self.writer = SummaryWriter(os.path.join(params.experiment_dir, 'tb_logs'))

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    best_acc1 = 0.
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)

      if epoch < params.lr_warmup_epochs:
        self.optimizer.param_groups[0]['lr'] = params.lr*float(epoch+1.)/float(params.lr_warmup_epochs)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()
      self.scheduler.step()

      is_best_acc1 = valid_logs['acc1'] > best_acc1
      best_acc1 = max(valid_logs['acc1'], best_acc1)

      if self.params.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_acc1)

      if self.params.log_to_tensorboard:
        self.writer.add_scalar('loss/train', train_logs['loss'], self.epoch) 
        self.writer.add_scalar('loss/valid', valid_logs['loss'], self.epoch) 
        self.writer.add_scalar('acc1/train', train_logs['acc1'], self.epoch) 
        self.writer.add_scalar('acc1/valid', valid_logs['acc1'], self.epoch) 
        self.writer.add_scalar('sigma/train', train_logs['sigma'], self.epoch) 
        self.writer.add_scalar('sigma/valid', valid_logs['sigma'], self.epoch) 
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)

      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info('train data time={}, train time={}, valid step time={}, train acc1={}, valid acc1={}'.format(data_time, tr_time,
                                                                                                                  valid_time,
                                                                                                                  train_logs['acc1'],
                                                                                                                  valid_logs['acc1']))

  def get_delzs(self, pdfs, speczs):
    bin_width = self.params.specz_upper_lim/self.params.num_classes
    span = (bin_width/2) + bin_width*torch.arange(0, self.params.num_classes)
    span = span.to(self.device)
    photozs = torch.sum((pdfs*span), axis = 1)
    delzs = (photozs-speczs)/(1+speczs)
    return delzs

  def compute_sigma_mad(self, delzs):
    madstd = mad_std(delzs)
    return np.float32(madstd)

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    self.model.train()
    softmax = torch.nn.Softmax(dim = 1)
    batch_size = self.params.batch_size # batch size per gpu
    n_samples = batch_size * len(self.train_data_loader)
    # pdfs/speczs for the samples on localgpu
    pdfs = torch.zeros(n_samples, self.params.num_classes).float().to(self.device)
    speczs = torch.zeros(n_samples).float().to(self.device)

    for i, data in enumerate(self.train_data_loader, 0):
      self.iters += 1
      data_start = time.time()
      images, specz_bin = map(lambda x: x.to(self.device), data[:2])
      specz = data[2].to(self.device)
      data_time += time.time() - data_start

      tr_start = time.time()
      self.model.zero_grad()
      with torch.cuda.amp.autocast(self.params.amp):
        outputs = self.model(images)
        loss = self.criterion(outputs, specz_bin)
      
      pdfs[i*batch_size:(i+1)*batch_size,:] = softmax(outputs).detach()
      speczs[i*batch_size:(i+1)*batch_size] = specz

      if self.params.amp:
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
      else:
        loss.backward()
        self.optimizer.step()
      tr_time += time.time() - tr_start
    
    delzs = self.get_delzs(pdfs, speczs) # a n_samples tensor on each gpu
    delzs_global = [torch.zeros(n_samples).float().to(self.device) for _ in range(params['world_size'])]
    dist.all_gather(delzs_global, delzs)
    delzs_global = torch.cat([dz for dz in delzs_global]).cpu().numpy()

    madstd = torch.from_numpy(np.array([self.compute_sigma_mad(delzs_global)])).to(self.device)

    # save metrics of last batch
    _, preds = outputs.max(1)
    acc1 = preds.eq(specz_bin).sum().float()/specz_bin.shape[0]
    logs = {'loss': loss,
            'acc1': acc1,
            'sigma': madstd}

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()

    valid_start = time.time()
    loss = 0.0
    correct = 0.0
    
    softmax = torch.nn.Softmax(dim = 1)
    batch_size = self.params.valid_batch_size_per_gpu
    n_samples = batch_size * len(self.valid_data_loader)
    # pdfs/speczs for the samples on localgpu
    pdfs = torch.zeros(n_samples, self.params.num_classes).float().to(self.device)
    speczs = torch.zeros(n_samples).float().to(self.device)

    with torch.no_grad():
      for idx, data in enumerate(self.valid_data_loader):
        images, specz_bin = map(lambda x: x.to(self.device), data[:2])
        specz = data[2].to(self.device)
        outputs = self.model(images)
        
        pdfs[idx*batch_size:(idx+1)*batch_size,:] = softmax(outputs).detach()
        speczs[idx*batch_size:(idx+1)*batch_size] = specz

        loss += self.criterion(outputs, specz_bin)
        _, preds = outputs.max(1)
        correct += preds.eq(specz_bin).sum().float()/specz_bin.shape[0]

    delzs = self.get_delzs(pdfs, speczs) # a n_samples tensor on each gpu
    delzs_global = [torch.zeros(n_samples).float().to(self.device) for _ in range(params['world_size'])]
    dist.all_gather(delzs_global, delzs)
    delzs_global = torch.cat([dz for dz in delzs_global]).cpu().numpy()

    madstd = torch.from_numpy(np.array([self.compute_sigma_mad(delzs_global)])).to(self.device)

    logs = {'loss': loss/len(self.valid_data_loader),
            'acc1': correct/len(self.valid_data_loader),
            'sigma': madstd}
    valid_time = time.time() - valid_start

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        logs[key] = torch.as_tensor(logs[key]).to(self.device)
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    return valid_time, logs

  def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    if is_best:
      torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path.replace('.tar', '_best.tar'))

  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
    self.model.load_state_dict(checkpoint['model_state'])
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch'] + 1
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", default=0, type=int)
  parser.add_argument("--yaml_config", default='./config/photoz.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
  parser.add_argument("--amp", action='store_true')
  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params['amp'] = args.amp

  # setup distributed training variables and intialize cluster if using
  params['world_size'] = 1
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])

  params['local_rank'] = args.local_rank
  params['world_rank'] = 0
  if params['world_size'] > 1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    params['world_rank'] = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  torch.backends.cudnn.benchmark = True

  # setup output directory
  expDir = os.path.join(*[args.root_dir, 'expts', args.config])
  if params.world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
      os.makedirs(os.path.join(expDir, 'checkpoints/'))

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'checkpoints/ckpt.tar')
  params['resuming'] = True if os.path.isfile(params.checkpoint_path) else False

  if params.world_rank==0:
    params.log()
  params['log_to_screen'] = params.log_to_screen and params.world_rank==0
  params['log_to_tensorboard'] = params.log_to_tensorboard and params.world_rank==0

  set_seed(params)
  trainer = Trainer(params)
  n_params = count_parameters(trainer.model)
  logging.info('number of model parameters: {}'.format(n_params))
  trainer.train()
