import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.anti_aliasing_downsample import Down

class CNN(nn.Module):
    def __init__(self, nf, factors=[2, 2, 2]) -> None:
        super().__init__()
        block = [
            nn.Conv2d(3, nf, 5, 1, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)            
        ]
        for _, f in enumerate(factors):
            block += [Down(nf, kernel_size=f+1, stride=f)]
            nf *= 2
            block += [ResBlock(dim=nf, dilation=1)]
            block += [ResBlock(dim=nf, dilation=3)]
        self.block = nn.Sequential(*block)        
    
    def forward(self, x):
        x = self.block(x)        
        return x

class TFAggregation(nn.Module):
    def __init__(self, emb_dim, ff_dim, n_heads, n_layers, p, tf_type) -> None:
        super().__init__()
        self.emb_dim = emb_dim        
        
        tf_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=p, activation=F.relu, layer_norm_eps=1e-5, batch_first=True)
        self.tf = nn.TransformerEncoder(tf_layer, num_layers=n_layers, norm=nn.LayerNorm(emb_dim),)          
        self.pos_emb = nn.Conv1d(emb_dim, emb_dim, kernel_size=7, stride=1, padding=3, padding_mode='zeros', groups=emb_dim, bias=True)        
        self.avg_pool = FastGlobalAvgPool(flatten=True)        
        
    def forward(self, x):                
        x = x + self.pos_emb(x)
        x = x.view(x.shape[0], self.emb_dim, -1).transpose(2, 1).contiguous()        
        x = self.tf(x)
        x = x.transpose(2, 1).contiguous()        
        return x

class Net(nn.Module):
    def __init__(self, emb_dim, n_classes, nf, factors, tf_type, inp_sz) -> None:
        super().__init__()
        self.nf = nf
        self.cnn = CNN(nf=16, factors=factors)
        self.tf = TFAggregation(emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1, tf_type=tf_type)                        
        self.project = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.tf(x)
        y = self.project(x)
        return y

if __name__ == "__main__":    
    pass