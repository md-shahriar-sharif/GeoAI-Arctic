import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
def _make_8band_conv(old_conv, pretrained):
    """Expand ResNet's RGB input layer to eight bands."""
    new_conv = nn.Conv2d(
        8,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            rgb_weights = old_conv.weight
            new_conv.weight[:, :3] = rgb_weights
            # Use the mean RGB filter to initialize the five extra bands.
            rgb_mean = rgb_weights.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = rgb_mean.repeat(1, 5, 1, 1)
        else:
            nn.init.kaiming_normal_(
                new_conv.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
    return new_conv
def build_8band_maskrcnn(pretrained=True):
    """Build the EXP002 8-band Mask R-CNN baseline."""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    # Adapt the backbone from RGB to the eight challenge bands.
    model.backbone.body.conv1 = _make_8band_conv(
        model.backbone.body.conv1,
        pretrained,
    )
    # Inputs are already standardized with training-fold statistics.
    model.transform.image_mean = [0.0] * 8
    model.transform.image_std = [1.0] * 8
    # Two classes: background and RTS.
    num_classes = 2
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_features,
        num_classes,
    )
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features,
        256,
        num_classes,
    )
    return model
