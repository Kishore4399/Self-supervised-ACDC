import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.transforms import Transform

class OneHotEncode(Transform):
    def __init__(self, keys, num_classes):
        """
        Args:
            num_classes (int): Number of classes for one-hot encoding.
        """
        self.keys = keys
        self.num_classes = num_classes

    def __call__(self, data):
        for key in self.keys:
            label = data[key]
            one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
            data[key] = one_hot.numpy()
        return data

class MakeChannelFirstd(Transform):
    def __init__(self, keys, channel_dim=-1):
        self.keys = keys
        self.channel_dim = channel_dim

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            data[key] = np.transpose(image, (self.channel_dim, 0, 1, 2))
        return data


class ChannelSelector(Transform):
    def __init__(self, keys, num_channels):
        """
        Args:
            keys (list): List of keys to apply the transform to.
            num_channels (int): Number of consecutive channels to select.
        """
        self.keys = keys
        self.num_channels = num_channels

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            total_channels = image.shape[0]
            
            if total_channels < self.num_channels:
                C, W, H, Z = image.shape
                image = torch.tensor(image, dtype=torch.float32)
                image = image.permute(3, 1, 2, 0)
                
                image = F.interpolate(image.unsqueeze(0), size=(W, H, self.num_channels), mode='trilinear', align_corners=True)
                image = image.squeeze(0).permute(3, 1, 2, 0)
                data[key] = image.numpy()
                continue

            start_idx = np.random.randint(0, total_channels - self.num_channels + 1)
            selected_image = image[start_idx:start_idx + self.num_channels, :, :, :]

            data[key] = selected_image
        return data

def get_mae_transform(args, type:str='train'):

    if type == 'train':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=['image']),
                transforms.AddChanneld(keys=['image']),
                transforms.Orientationd(keys=['image'],
                                        axcodes="RAS"),

                transforms.Resized(keys=['image'], 
                                   spatial_size=(args.resize_x, args.resize_y, args.resize_z)),
                transforms.RandSpatialCropd(keys=['image'],
                                            roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                            random_size=False),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=['image'], dtype=torch.float32),
            ]
        )
        return train_transform
    
    elif type == 'valide':
        valide_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=['image']),
                transforms.AddChanneld(keys=['image']),
                transforms.Orientationd(keys=['image'],
                                        axcodes="RAS"),
                transforms.Resized(keys=['image'], 
                                   spatial_size=(args.resize_x, args.resize_y, args.resize_z)),
                transforms.CenterSpatialCropd(
                                    keys=['image'],
                                    roi_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=['image'], dtype=torch.float32),
            ]
        )
        return valide_transform
    

def get_vit4d_transform(args, type:str='train'):

    if type == 'train':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=['image']),
                MakeChannelFirstd(keys=['image'], channel_dim=3),
                transforms.Orientationd(keys=['image'],
                                        axcodes="RAS"),
                ChannelSelector(keys=['image'], num_channels=args.roi_t),
                transforms.Resized(keys=['image'], 
                                   spatial_size=(args.resize_x, args.resize_y, args.resize_z)),
                transforms.RandSpatialCropd(keys=['image'],
                                            roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                            random_size=False),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=['image'],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
                transforms.ToTensord(keys=['image', 'class']),
            ]
        )
        return train_transform
    
    elif type == 'valide':
        valide_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=['image']),
                MakeChannelFirstd(keys=['image'], channel_dim=3),
                transforms.Orientationd(keys=['image'],
                                        axcodes="RAS"),
                ChannelSelector(keys=['image'], num_channels=args.roi_t),
                transforms.Resized(keys=['image'], 
                                   spatial_size=(args.resize_x, args.resize_y, args.resize_z)),
                transforms.CenterSpatialCropd(
                                    keys=['image'],
                                    roi_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
                transforms.ToTensord(keys=['image', 'class']),
            ]
        )
        return valide_transform