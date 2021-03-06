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
    """

    def __init__(self, w_pos):
        super().__init__()
        self.w_pos = w_pos

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
        # object detection loss
        obj_loss = F.binary_cross_entropy_with_logits(cls_output[:, 0],
            cls_target[:, 0].float(), pos_weight=torch.tensor(self.w_pos,
            device=cls_output.device))
        # object classification loss
        pos_indices = torch.where(cls_target[:, 0] == 1)
        clf_loss = F.cross_entropy(
            cls_output[pos_indices[0], 1:, pos_indices[1], pos_indices[2]],
            cls_target[pos_indices[0], 1, pos_indices[1], pos_indices[2]]
        )

        cls_loss = clf_loss + obj_loss

        return cls_loss


class DetectLoss(nn.Module):
    """
    Detection loss.

    Parameters
    ----------
    w_cls : float
        Weight for classification loss.
    w_reg : float
        Weight for regression loss.
    w_pos : float
        Weight for positive samples.
    w_neg : float
        Weight for negative samples.
    """

    def __init__(self, w_cls, w_reg, w_pos):
        super().__init__()
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_pos = w_pos

        self.cls_loss_fn = ClsCrossEntropy(self.w_pos)
        self.reg_loss_fn = RegSmoothL1()

    def forward(self, cls_output, reg_output, cls_target, reg_target):
        """
        Parameters
        ----------
        cls_output : torch.Tensor
            Classification output.
        reg_output : torch.Tensor
            Regression output.
        cls_target : torch.Tensor
            Classification target.
        reg_target : torch.Tensor
            Regression target.

        Returns
        -------
        total_loss : torch.Tensor
            Total detection loss combining classification and regression loss.
        cls_loss : torch.Tensor
            Classification loss.
        reg_loss : torch.Tensor
            Regression loss.
        """
        cls_loss = self.cls_loss_fn(cls_output, cls_target)
        reg_loss = self.reg_loss_fn(reg_output, reg_target)
        total_loss = self.w_cls * cls_loss + self.w_reg * reg_loss

        return total_loss, cls_loss, reg_loss
