import torch.nn as nn
from torchvision.models import resnext50_32x4d


def _get_resnext_backbone(backbone_fn, pretrained):
    """
    Get ResNeXt backbone.
    """
    model = backbone_fn(pretrained)
    backbone = nn.Sequential()

    for name, layer in model.named_children():
        if name == "avgpool":
            break

        backbone.add_module(name, layer)
    
    return backbone


class _OutputHead(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, 1))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("actv", nn.ReLU())


class YOLOResNeXt(nn.Module):
    """
    ResNeXt based YOLO.

    Parameters
    ----------
    backbone_fn : function
        Function tht returns a PyTorch model.
    num_classes : int
        Number of classes.
    pretrained : bool
        Whether to load pretrained weights.
    """

    def __init__(self, backbone_fn, num_classes, pretrained=True):
        super().__init__()
        self.backbone = _get_resnext_backbone(backbone_fn, pretrained)
        self.cls_head = _OutputHead(2048, 1 + num_classes)
        self.reg_head = _OutputHead(2048, 4)

    def forward(self, images):
        """
        Parameters
        ----------
        images : torch.Tensor
            Image tensor in shape of N x H x W x C.
        
        Returns
        -------
        cls_output : torch.Tensor
            Classification output.
        reg_output : torch.Tensor
            Regression output.
        """
        features = self.backbone(images)
        cls_output = self.cls_head(features)
        reg_output = self.reg_head(features)

        return cls_output, reg_output
