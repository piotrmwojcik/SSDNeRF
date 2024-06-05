import torch.nn as nn
import torch
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def reg_loss(tensor, power=1):
    _, code_3p = torch.split(tensor, [5, 1], dim=2)
    return code_3p.abs().mean() if power == 1 \
        else (code_3p.abs() ** power).mean()


@MODULES.register_module()
class RegLoss(nn.Module):

    def __init__(self,
                 power=1,
                 loss_weight=1.0):
        super().__init__()
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, tensor, weight=None, avg_factor=None):
        return reg_loss(
            tensor, power=self.power,
            weight=weight, avg_factor=avg_factor) * self.loss_weight
