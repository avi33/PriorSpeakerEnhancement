import torch
import torch.nn as nn

class SISDRLoss(nn.Module):
    def __init__(self, eps=1e-8, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
    
    def forward(self, estimate, reference):
        estimate = estimate - estimate.mean(dims=-1)
        reference = reference - reference.mean(dims=-1)
        reference_pow = (reference**2).mean(dims=-1, keepdim=True)
        mix_pow = (estimate * reference).mean(dims=-1, keepdim=True)
        scale = mix_pow / (reference_pow + self.eps)
        reference = scale * reference
        error = estimate - reference

        reference_pow = reference**2
        error_pow = error**2

        reference_pow = reference_pow.mean(dims=-1)
        error_pow = error_pow.mean(dims=-1)
        si_sdr = 10*torch.log10(reference_pow) - 10*torch.log10(error_pow)
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            si_sdr = si_sdr.sum()
        elif self.reduction == "mean":
            si_sdr = si_sdr.mean()
        else:
            NotImplementedError(self.reduction)
        return -si_sdr