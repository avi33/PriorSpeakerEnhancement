import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
class Downsample2dJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0, device: str = "cuda"):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        # assert self.filt_size == 3
        # assert stride == 2
        
        device = torch.device(device)
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        # a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).to(device).half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, [self.filt_size//2]*4, 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])

class Downsample2d(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None, device="cuda"):
        super(Downsample2d, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        # assert self.filt_size == 3
        device = torch.device(device)
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()


        a = torch.tensor([1., 2., 1.])

        filt = (a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).to(device).half()
        self.register_buffer('filt', filt)

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])

class AntiAliasDownsample2dLayer(nn.Module):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0, device: str = "cuda"):
        super(AntiAliasDownsample2dLayer, self).__init__()
        if not remove_aa_jit:
            self.op = Downsample2dJIT(filt_size, stride, channels, device=device)
        else:
            self.op = Downsample2d(filt_size, stride, channels, device=device)

    def forward(self, x):
        return self.op(x)

class AntiAliasDownsample1dLayer(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AntiAliasDownsample1dLayer, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size//2+1+1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1,])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size//2, self.filt_size//2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y

class Down2d(nn.Module):
    def __init__(self, nf, kernel_size, stride) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size//2, padding_mode="reflect"),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            AntiAliasDownsample2dLayer(channels=nf*2, stride=stride, filt_size=kernel_size)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Down1d(nn.Module):
    def __init__(self, nf, kernel_size, stride) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(nf, nf*2, kernel_size=kernel_size, stride=1, bias=False, padding=kernel_size//2, padding_mode="reflect"),
            nn.BatchNorm1d(nf*2),
            nn.LeakyReLU(0.2, True),
            AntiAliasDownsample1dLayer(channels=nf*2, stride=stride, filt_size=kernel_size)
        )

    def forward(self, x):
        x = self.block(x)
        return x