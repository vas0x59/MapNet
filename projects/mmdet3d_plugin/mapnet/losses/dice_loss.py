import torch
import torch
import torch.nn as nn

from mmdet.models.losses.utils import weighted_loss
from mmdet.models.builder import LOSSES

@weighted_loss
def dice_loss(input, target,mask=None,eps=0.001):
    N, C, H, W = input.shape  # Размерность батча и каналов

    # Преобразуем в 2D: (N, C, H*W)
    input = input.contiguous().view(N, C, -1)
    target = target.contiguous().view(N, C, -1).float()

    if mask is not None:
        mask = mask.contiguous().view(N, C, -1).float()
        input = input * mask
        target = target * mask

    # Вычисляем компоненты Dice Loss
    intersection = torch.sum(input * target, dim=2)  # Скалярное произведение по пикселям
    input_sum = torch.sum(input, dim=2) + eps  # Сумма предсказаний
    target_sum = torch.sum(target, dim=2) + eps  # Сумма истинных значений

    dice_coeff = (2 * intersection) / (input_sum + target_sum)  # Dice Coefficient
    loss = 1 - dice_coeff  # Dice Loss

    return loss.mean()  # Усредняем по батчу и каналам

@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0
    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred = torch.sigmoid(pred)
        #if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n,w,h) to (n,) to match the
            # giou_loss of shape (n,)
            #assert weight.shape == pred.shape
            #weight = weight.mean((-2,-1))
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            mask=mask,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        #print('DiceLoss',loss, avg_factor)
        return loss
