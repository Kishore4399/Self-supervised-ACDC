import os
import math
import numpy as np
import pickle

import time
import torch
import torch.nn.functional as F
from monai import transforms

import sys
sys.path.append('..')

import models
import networks
from utils import SmoothedValue, concat_all_gather, compute_aucs
from monai.data import CacheDataset, Dataset, DataLoader
from .base_trainer import BaseTrainer
from data.med_transforms import get_mae_transform
from tools.visualization import patches3d_to_grid

import wandb

class MAE3DTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "MAE3D"
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch),
                            decoder=getattr(networks, args.dec_arch), 
                            args=args)
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
        
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")
        
    def build_optimizer(self):
        assert(self.model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)
            
        
    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args
            pkl_path = os.path.join(args.data_path, args.pkl_list)
            with open(pkl_path, 'rb') as file:
                loaded_dic = pickle.load(file)

            train_ds = []
            for dic_tr in loaded_dic['training']:
                dic_tr['image'] = os.path.join(args.data_path, dic_tr['image'])
                train_ds.append(dic_tr)

            val_ds = []
            for dic_vl in loaded_dic['validating']:
                dic_vl['image'] = os.path.join(args.data_path, dic_vl['image'])
                val_ds.append(dic_vl)


            train_transform = get_mae_transform(args, 'train')
            train_dataset = CacheDataset(train_ds,
                                          transform=train_transform,
                                          num_workers=args.workers,
                                          cache_num=len(loaded_dic['training']))

            val_transform = get_mae_transform(args, 'valide')
            val_dataset = Dataset(val_ds, transform=val_transform)
            
            self.dataloader = DataLoader(train_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=self.workers,
                                         pin_memory=True,
                                         sampler=None,
                                         drop_last=True)
            
            self.iters_per_epoch = len(self.dataloader)
            
            self.val_dataloader = DataLoader(val_dataset, 
                                            batch_size=args.val_batch_size, 
                                            shuffle=True,
                                            num_workers=self.workers, 
                                            pin_memory=True, 
                                            sampler=None,
                                            drop_last=False)
            self.val_iters = len(self.val_dataloader)
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")

    def run(self):
        args = self.args
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            niters = self.epoch_train(epoch, niters)

            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                print(f"=> start visualizing after {epoch + 1} epochs")
                self.vis_reconstruction(niters)
                print("=> finish visualizing")
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(), # additional line compared with base imple
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar')
                    print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # For SSL pretraining, only image data is required for training
            image = batch_data['image']

            if self.device.type == "cuda":
                image = image.to(self.device)

            # compute output and loss
            forward_start_time = time.time()
            with torch.cuda.amp.autocast(True):
                loss = model(image, return_image=False)
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # Log to the screen
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
            load_start_time = time.time()
        return niters


    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
        
    @torch.no_grad()
    def vis_reconstruction(self, niters=0):
        args = self.args
        loader = self.val_dataloader
        model = self.model

        model.eval()

        for batch_data in loader:
            image = batch_data['image']
            if self.device.type == "cuda":
                image = image.to(self.device)

            # compute output and loss
            _, x, recon, masked_x = model(image, return_image=True)

            vis_tensor = torch.cat([x, masked_x, recon], dim=0)

            # visualize
            grid_size = []
            input_size = tuple([args.roi_x, args.roi_y, args.roi_z])
            patch_size = tuple([args.patch_x, args.patch_y, args.patch_z])
            for pa_size, in_size in zip(patch_size, input_size):
                grid_size.append(in_size // pa_size)
            vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=patch_size, grid_size=tuple(grid_size), in_chans=args.in_chans, hidden_axis='d')
            # import pdb
            # pdb.set_trace()
            # vis_grid_hd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='w')
            # vis_grid_wd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='h')

            print("wandb logging")
            vis_grid_hw = wandb.Image(vis_grid_hw, caption=f"hw_iter{niters:06d}")
            # vis_grid_hd = wandb.Image(vis_grid_hd, caption=f"hd_iter{niters:06d}")
            # vis_grid_wd = wandb.Image(vis_grid_wd, caption=f"wd_iter{niters:06d}")

            wandb.log(
                {
                "vis_hw": vis_grid_hw,
                # "vis_hd": vis_grid_hd,
                # "vis_wd": vis_grid_wd
                },
                step=niters,
            )
            break
        print("finish wandb logging")


    
