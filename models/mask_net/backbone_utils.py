from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
from ._utils import IntermediateLayerGetter
from .. import resnet
#import resnet

class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            # Alex: delete the additional block
            # FPN outputs a single layer
            #extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


def resnet_fpn_backbone(backbone_name, pretrained, out_ch, truncation):

    
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=nn.BatchNorm2d, truncation=truncation)
    # Alex: the orginial implementation uses 4 outputs from the backbone
    # to replicate the best results in the paper, use resnet18 or resnet34
    # and truncation=1 or resnet50 and truncation=0 for the large model
    if backbone_name == 'resnet50':
       return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
       in_channels_stage2 = 256
       in_channels_list = [
          in_channels_stage2,
          in_channels_stage2 * 2,
          in_channels_stage2 * 4,
          in_channels_stage2 * 8,
      ]

    elif backbone_name == 'resnet18' or backbone_name == 'resnet34':
       if truncation == '0':
          return_layers = {'layer4':0}
       elif truncation == '1':
          return_layers = {'layer3':0}
       elif truncation == '2':
          return_layers = {'layer2':0}
       # Alex: I added the feature that returns the number of channels from the last layer in the net
       in_channels_list = [backbone.out_channels]

    # These should be 256
    out_channels = out_ch
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
