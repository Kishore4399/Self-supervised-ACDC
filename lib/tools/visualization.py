import numpy as np

import torch
import torchvision

def patches3d_to_grid(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, L, C = patches.shape
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"
    # print(f"grid_size: {grid_size[0]}, patch_size: {patch_size[0]}")

    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)
    # restore image structure
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0], 
                                                            grid_size[1] * patch_size[1], 
                                                            grid_size[2] * patch_size[2])

    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
        image = torch.cat(slice_list, dim=2)
    else:
        raise ValueError(f"Only support D for now")
    visH, visW = image.size(-2), image.size(-1)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)
    # grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return grid_of_images