# -*- coding: utf-8 -*-
# Dummy implementation for inference - this is only needed for training

import torch
import torch.nn as nn


class FusedLinearDiffusionCrossEntropyLoss(nn.Module):
    """
    Dummy implementation for inference.
    This loss function is only used during training, not inference.
    """
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, labels, **kwargs):
        # During inference, we don't compute loss, so return dummy
        if self.reduction == 'sum':
            return logits.sum() * 0.0
        elif self.reduction == 'mean':
            return logits.mean() * 0.0
        else:
            return logits * 0.0
