import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_levels, pool = True):
        super(ConvBlock, self).__init__()

        modules = []

        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias = False), nn.BatchNorm3d(out_channels), nn.ReLU())
        if pool:
            self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2,))
        else:
            self.pool = None

        for i in range(n_levels):
            layer = nn.Sequential(nn.Conv3d(out_channels * (i + 1), out_channels, 3, 1, 1, bias = False), 
            nn.BatchNorm3d(out_channels), 
            nn.ReLU())
            modules.append(layer)

        self.layers = nn.ModuleList(modules)

        self.final = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, 2, 1, bias = False), nn.BatchNorm3d(out_channels), nn.ReLU())

    def forward(self, x):

        out = self.conv(x)

        if self.pool is not None:
            out = self.pool(out)

        prev = out

        for indx, layer in enumerate(self.layers):
            new = layer(prev)
            prev = torch.cat([prev, new], 1)

        out = new + out

        del prev

        return self.final(out)


class Model(nn.Module):

    def __init__(self, in_channels, out_channels, hid_dims, n_levels):

        super(Model, self).__init__()


        self.conv = torch.nn.Conv3d(in_channels, hid_dims[0], 3, 1, 1)

        blocks = []

        for i in range(len(hid_dims)):
            if i == len(hid_dims) - 1:
                block = ConvBlock(hid_dims[i], out_channels, n_levels, pool = False)
            else:
                block = ConvBlock(hid_dims[i], hid_dims[i + 1], n_levels)

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Conv3d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):

        B, C, H, W, D = x.shape

        out = self.conv(x)

        for layer in self.blocks:
            out = layer(out)
            
        return self.final(out).view(B, -1)

if __name__ == '__main__':

    model = Model(1, 10, [32, 64, 10], 2).cuda()
    i = torch.randn(8, 1, 3, 32, 32)



