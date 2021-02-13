import torch.nn as nn
from torchvision.models import resnext50_32x4d


def get_resnext_backbone(backbone_fn, pretrained=True):
    """
    Get ResNeXt backbone.

    Parameters
    ----------
    backbone_fn : function
        Function tht returns a PyTorch model.
    pretrained : bool, optional
        Whether to load pretrained weights. The default value is 0.
    
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


if __name__ == "__main__":
    model = _get_resnext_backbone(resnext50_32x4d)
    print(model)
