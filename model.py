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


class YOLOResNeXt(nn.Sequential):

    def __init__(self, backbone_fn, num_classes, pretrained=True):
        super().__init__()
        self.add_module("backbone", _get_resnext_backbone(backbone_fn,
            pretrained=pretrained))
        self.add_module("output_layer", nn.Conv2d(2048, num_classes, 1)) 


if __name__ == "__main__":
    from torchsummary import summary


    model = YOLOResNeXt(resnext50_32x4d, 21).cuda()
    summary(model, (3, 224, 224))
