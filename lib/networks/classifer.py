import torch.nn as nn
import numpy as np

class CNNDecoder(nn.Module):
    def __init__(self, args, channel_sizes, embed_dim, num_classes, kernel_size=3, pool_kernel_size=2):
        super(CNNDecoder, self).__init__()
        
        layers = []
        layers.append(nn.Conv1d(args.roi_t, channel_sizes[0], kernel_size, padding=1))
        layers.append(nn.ReLU())

        for i in range(len(channel_sizes) - 1):
            in_channels = channel_sizes[i]
            out_channels = channel_sizes[i + 1]
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_kernel_size))
        
        conv_output_length = embed_dim
        for _ in range(len(channel_sizes) - 1):
            conv_output_length = (conv_output_length + 1) // 2

        self.conv_block = nn.Sequential(*layers)
        self.fc1 = nn.Linear(conv_output_length * channel_sizes[-1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output