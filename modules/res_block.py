import torch.nn as nn

class ResBlock2d(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResBlock2d, self).__init__()
        self.block = nn.Sequential(            
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, kernel_size=3, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, kernel_size=1),            
        )
        self.shortcut = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResBlock1d(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResBlock1d, self).__init__()
        self.block = nn.Sequential(            
            nn.ReflectionPad1d(dilation),
            nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, dim, kernel_size=1),            
        )
        self.shortcut = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)