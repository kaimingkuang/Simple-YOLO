import torch.nn as nn
from torchvision.models import resnext50_32x4d


def _get_resnext_backbone(backbone_fn, pretrained):
    """
    Get ResNeXt backbone.

    Parameters
    ----------
    backbone_fn : function
        Function tht returns a PyTorch model.
    pretrained : bool
        Whether to load pretrained weights.
    
    Returns
    -------
    backbone : nn.Sequential
        A PyTorch model containing only layers before GAP.
    """
    model = backbone_fn(pretrained)
    backbone = nn.Sequential()

    for name, layer in model.named_children():
        if name == "avgpool":
            break

        backbone.add_module(name, layer)
    
    return backbone


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
        self.cls_head = nn.Conv2d(2048, num_classes, 1)
        self.reg_head = nn.Conv2d(2048, 4, 1)

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
