import torch
import torch.nn as nn

class MaskedMSELoss(nn.MSELoss):
    def forward(self, input, target, mask=None):
        if mask is not None:
            input = input[:, :-1][mask]
            target = target[mask]
        return super().forward(input, target)
