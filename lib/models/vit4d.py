import torch
import torch.nn as nn
import numpy as np

__all__ = ["ViTCNN4D"]

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed

def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2], patch_size[2]) # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size), np.prod(patch_size) * C) # [B,gh*gw*gd,ph*pw*pd*C]

    return x

class ViTCNN4D(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 args):
        super().__init__()
        self.args = args

        input_size = tuple([args.roi_x, args.roi_y, args.roi_z])
        patch_size = tuple([args.patch_x, args.patch_y, args.patch_z])
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        
        self.grid_size = grid_size

        with torch.no_grad():
            self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                        args.encoder_embed_dim, 
                                                                        num_tokens=0)

        # build encoder and decoder
        from lib.networks import patch_embed_layers
        embed_layer = getattr(patch_embed_layers, args.patchembed)(patch_size, 
                                                                   patch_size, 
                                                                   args.in_chans, 
                                                                   args.encoder_embed_dim)

        self.encoder = encoder(embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        
        self.decoder = decoder(args=args,
                                embed_dim=args.decoder_embed_dim,
                                channel_sizes=args.channel_sizes,
                                num_classes=args.num_classes)

        self.encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)

        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.view(B*C, self.args.in_chans, X, Y, Z)

        x = patchify_image(x, self.patch_size)
        pos_embed = self.encoder_pos_embed.expand(B*C, -1, -1)
        x = self.encoder(x, pos_embed) # (B*C, N, D1) 
        x = x.view(B, C, -1)  # (B, C, N*D1) 
        x = self.encoder_to_decoder(x)  # (B, C, D2) 
        output = self.decoder(x)
        return output