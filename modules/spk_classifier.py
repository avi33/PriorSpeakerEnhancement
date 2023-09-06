import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.anti_aliasing_downsample import Down1d
from modules.res_block import ResBlock1d

class CNN(nn.Module):
    def __init__(self, c_in, nf, factors=[2, 2, 2]) -> None:
        super().__init__()
        block = [
            nn.Conv1d(c_in, nf, 5, 1, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True)            
        ]
        for _, f in enumerate(factors):
            block += [Down1d(nf, kernel_size=f+1, stride=f)]
            nf *= 2
            block += [ResBlock1d(dim=nf, dilation=1)]
            block += [ResBlock1d(dim=nf, dilation=3)]
        self.block = nn.Sequential(*block)        
    
    def forward(self, x):
        x = self.block(x)        
        return x

class TFAggregation(nn.Module):
    def __init__(self, emb_dim, ff_dim, n_heads, n_layers, p) -> None:
        super().__init__()
        self.emb_dim = emb_dim        
        
        tf_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=p, activation=F.relu, layer_norm_eps=1e-5, batch_first=True)
        self.tf = nn.TransformerEncoder(tf_layer, num_layers=n_layers, norm=nn.LayerNorm(emb_dim),)          
        self.pos_emb = nn.Conv1d(emb_dim, emb_dim, kernel_size=7, stride=1, padding=3, padding_mode='zeros', groups=emb_dim, bias=True)
        
    def forward(self, x):                
        x = x + self.pos_emb(x)
        x = x.view(x.shape[0], self.emb_dim, -1).transpose(2, 1).contiguous()        
        x = self.tf(x)
        x = x.transpose(2, 1).contiguous()        
        return x

class SpkEncoder(nn.Module):
    def __init__(self, kwargs) -> None:        
        super().__init__()
        self.fft_params = kwargs['fft_params']
        self.nf = kwargs['nf']
        self.cnn = CNN(c_in=2*(self.fft_params['n_fft']//2+1), nf=self.nf, factors=kwargs['factors'])
        self.tf = TFAggregation(emb_dim=kwargs['emb_dim'], ff_dim=kwargs['emb_dim']*4, n_heads=2, n_layers=4, p=0.1)        

    def forward(self, x, emb_dim):
        n_pad = (self.fft_params['win_length'] - self.fft_params['hop_length']) // 2
        xp = F.pad(x, (n_pad, n_pad))
        X = torch.stft(xp, **self.fft_params)        
        X = X.unsqueeze(2)
        X = torch.cat((X.real, X.imag), dim=2).contiguous()
        X = X.view(X.shape[0], -1, X.shape[-1]).contiguous()
        X = self.cnn(X)
        X = self.tf(X)
        return X

if __name__ == "__main__":    
    pass