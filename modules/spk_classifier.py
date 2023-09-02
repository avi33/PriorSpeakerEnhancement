import torch
import torch.nn as nn

class SpkEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    
    def forward(self, x):
        return x