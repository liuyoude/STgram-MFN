import torch
import torch.nn as nn


class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss