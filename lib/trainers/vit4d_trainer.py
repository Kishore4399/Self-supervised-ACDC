import os
import math
import pickle
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import CacheDataset, Dataset, DataLoader

import sys
sys.path.append('..')

import models
import networks
from utils import SmoothedValue, concat_all_gather, compute_aucs
from .base_trainer import BaseTrainer
from data.med_transforms import get_vit4d_transform

import wandb

from timm.data.mixup import Mixup
from timm.utils import accuracy

from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class ViT4DTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ViTCNN4D"
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args

            # setup mixup and loss functions
            if args.mixup > 0:
                self.mixup_fn = Mixup(
                            mixup_alpha=args.mixup, 
                            cutmix_alpha=args.cutmix, 
                            label_smoothing=args.label_smoothing, 
                            num_classes=args.num_classes)             
            else:
                self.mixup_fn = None
            
            if args.loss_fn == "cel":
                self.loss_fn = nn.CrossEntropyLoss()
            elif args.loss_fn == "bcelog":
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                raise ValueError("=> Undefined loss function")


            self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch), 
                            decoder=getattr(networks, args.dec_arch), 
                            args=args)
            
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')

            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
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


            train_transform = get_vit4d_transform(args, 'train')
            train_dataset = CacheDataset(train_ds,
                                          transform=train_transform,
                                          num_workers=args.workers,
                                          cache_num=len(loaded_dic['training']))

            val_transform = get_vit4d_transform(args, 'valide')
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

            # evaluate after each epoch training
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                acc1_global = self.evaluate(epoch=epoch, niters=niters)
            
                if (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False,
                        filename=f'{args.ckpt_dir}/{args.run_name}_acc{acc1_global:.02f}_checkpoint_{epoch:04d}.pth.tar'
                    )
                    del_files = sorted(glob.glob(f'{args.ckpt_dir}/{args.run_name}*.pth.tar'), key= lambda x: os.stat(x).st_mtime)[:int(-1*args.save_ckpt_num)]

                    if len(del_files) > 0:
                        for del_file_path in del_files:
                            os.remove(del_file_path)
            
                       

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        loss_fn = self.loss_fn

        model.train()

        for i, input_batch in enumerate(train_loader):
            image = input_batch['image']
            target = input_batch['class']
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            if self.device.type == "cuda":
                image = image.to(self.device)
                target = target.to(self.device)
                if args.loss_fn == "bcelog":
                    target = F.one_hot(torch.tensor(target), num_classes=args.num_classes).to(torch.float32)

                with torch.cuda.amp.autocast(True):
                    loss = self.train_class_batch(model, image, target, loss_fn)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      # f"PeRate: {model.module.pe_rate:.05f} | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": last_layer_lr,
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
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

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = model(samples)
        loss = criterion(outputs, target)
        return loss
        
    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.model
        val_loader = self.val_dataloader
        if args.eval_metric == 'acc':
            criterion = torch.nn.CrossEntropyLoss()
        elif args.eval_metric == 'auc':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Only support acc and auc for now")
        meters = defaultdict(SmoothedValue)

        # switch to evaluation mode
        model.eval()

        if args.eval_metric == 'auc':
            pred_list = []
            targ_list = []

        for i, input_batch in enumerate(val_loader):
            image = input_batch['image']
            target = input_batch['class']
            if self.device.type == "cuda":
                image = image.to(self.device)
                target = target.to(self.device)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output, target)
            

            if args.eval_metric == 'acc':
                acc1, acc2 = accuracy(output, target, topk=(1, 2))
   
                batch_size = image.size(0)
                meters['loss'].update(value=loss.item(), n=batch_size)
                meters['acc1'].update(value=acc1.item(), n=batch_size)
                meters['acc2'].update(value=acc2.item(), n=batch_size)

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.val_iters} | "
                      f"Loss: {loss.item():.03f} | "
                      f"Acc1: {acc1.item():.03f} | "
                      f"Acc2: {acc2.item():.03f} | ")
            elif args.eval_metric == 'auc':
                batch_size = image.size(0)
                meters['loss'].update(value=loss.item(), n=batch_size)
                pred_list.append(concat_all_gather(output, args.distributed))
                targ_list.append(concat_all_gather(target, args.distributed))

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.val_iters} | "
                      f"Loss: {loss.item():.03f}")
            else:
                raise NotImplementedError("Only support Acc and AUC for now.")
        
        # compute auc
        if args.eval_metric == 'auc':
            pred_array = torch.cat(pred_list, dim=0).data.cpu().numpy()
            targ_array = torch.cat(targ_list, dim=0).data.cpu().numpy()
            auc_list, mAUC = compute_aucs(pred_array, targ_array)

            print(f"==> Epoch {epoch:04d} test results: \n"
                  f"=> mAUC: {mAUC:.05f}")

            if args.rank == 0:
                wandb.log(
                    {
                     "Eval Loss": meters['loss'].global_avg,
                     "mAUC": mAUC
                    },
                    step=niters,
                )
        elif args.eval_metric == 'acc':
            print(f"==> Epoch {epoch:04d} test results: \n"
                  f"=>  Loss: {meters['loss'].global_avg:.05f} \n"
                  f"=> Acc@1: {meters['acc1'].global_avg:.05f} \n"
                  f"=> Acc@2: {meters['acc2'].global_avg:.05f} \n")
            
            acc1_global = meters['acc1'].global_avg

            if args.rank == 0:
                wandb.log(
                    {
                     "Eval Loss": meters['loss'].global_avg,
                     "Acc@1": meters['acc1'].global_avg,
                     "Acc@2": meters['acc2'].global_avg
                    },
                    step=niters,
                )
            return acc1_global