import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as f

################################### LOSS ###################################

def tversky_loss(y_pred, y_true, smooth=1, alpha=0.3, beta=0.7):
  y_pred = y_pred.permute(1, 2, 3, 0).contiguous().reshape(3, -1)
  y_true = f.one_hot(y_true.squeeze(1).long(), num_classes=3).permute(3, 1, 2, 0).contiguous().reshape(3, -1) # C x H x W x N
  tp = (y_pred[1:] * y_true[1:]).sum(dim=1)
  fp = (y_pred[1:] * (1 - y_true[1:])).sum(dim=1)
  fn = ((1 - y_pred[1:]) * y_true[1:]).sum(dim=1)

  tversky_class = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
  return (1 - tversky_class) # C x 1

def focal_tversky_loss(y_pred, y_true, gamma=3/4):
    return torch.mean(torch.sum(torch.pow(tversky_loss(y_pred, y_true), gamma)))

def multiclass_loss(y_pred, y_true):
    ft_loss = focal_tversky_loss(y_pred, y_true)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, 3)
    ce_loss = nn.CrossEntropyLoss()(y_pred, y_true.view(-1).long())
    return 0.5 * ce_loss + 0.5 * ft_loss


################################### METRIC ###################################

def iou_dice_score(y_pred, y_true, smooth=1):
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = f.one_hot(y_pred, num_classes=3).permute(3, 1, 2, 0).contiguous().reshape(3, -1)
    y_true = f.one_hot(y_true.squeeze(1).long(), num_classes=3).permute(3, 1, 2, 0).contiguous().reshape(3, -1) # C x H x W x N
    tp = (y_pred[1:] * y_true[1:]).sum(dim=1)
    fp = (y_pred[1:] * (1 - y_true[1:])).sum(dim=1)
    fn = ((1 - y_pred[1:]) * y_true[1:]).sum(dim=1)

    iou_class = (tp + smooth) / (tp + fp + fn  + smooth)
    dice_class = (2 * tp + smooth) / (2 * tp + fp + fn  + smooth)
    iou = torch.mean(iou_class)
    dice = torch.mean(dice_class)

    return iou, dice