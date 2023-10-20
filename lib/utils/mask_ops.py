import torch
import torch.nn as nn
# Here we will build a toolbox used to manipulate the segmentation masks

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        sum_pred = torch.sum(y_pred)
        sum_true = torch.sum(y_true)
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        loss = 1 - iou
        return loss