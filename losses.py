import torch
import torch.nn as nn
import torch.nn.functional as F


NINF = -9e6


class RegSmoothL1(nn.Module):
    """
    Smooth L1 regression loss.
    """

    def forward(self, reg_output, reg_target):
        """
        Parameters
        ----------
        reg_output : torch.Tensor
            Regression output.
        reg_target : torch.Tensor
            Regression target.
        
        Returns
        -------
        loss : torch.Tensor
            Regression loss.
        """
        pos_reg_target = reg_target[reg_target > NINF].reshape(-1)
        pos_reg_output = reg_output[reg_target > NINF].reshape(-1)
        reg_loss = F.smooth_l1_loss(pos_reg_output, pos_reg_target)

        return reg_loss


class ClsCrossEntropy(nn.Module):
    """
    Classification cross entropy loss with positive-negative balanced weights.

    Parameters
    ----------
    w_pos : float
        Weight for positive samples.
    w_neg : float
        Weight for negative samples.
    """

    def __init__(self, w_pos, w_neg):
        super().__init__()
        self.w_pos = w_pos
        self.w_neg = w_neg
    
    def forward(self, cls_output, cls_target):
        """
        Parameters
        ----------
        cls_output : torch.Tensor
            Classification output.
        cls_target : torch.Tensor
            Classification target.

        Returns
        -------
        cls_loss : torch.Tensor
            Classification loss.
        """
        pos_loss = F.cross_entropy(cls_output, cls_target, ignore_index=0)
        cls_target = torch.where(cls_target > 0, 1, cls_target)
        neg_loss = F.cross_entropy(cls_output, cls_target, ignore_index=1)
        cls_loss = self.w_pos * pos_loss + self.w_neg * neg_loss

        return cls_loss


if __name__ == "__main__":
    cls_target = torch.tensor([0, 1, 2, 3])
    cls_output = torch.rand(4, 4)
    print(F.cross_entropy(cls_output, cls_target, ignore_index=3,
        reduction="none"))
