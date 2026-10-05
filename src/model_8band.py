import torch

import torch.nn as nn



from torchvision.models.detection import (

    maskrcnn_resnet50_fpn,

    MaskRCNN_ResNet50_FPN_Weights,

)



from torchvision.models.detection.faster_rcnn import (

    FastRCNNPredictor,

)



from torchvision.models.detection.mask_rcnn import (

    MaskRCNNPredictor,

)





def build_8band_maskrcnn(pretrained=True):

    """

    EXP002: 8-band Mask R-CNN.



    Architecture:

        Mask R-CNN + ResNet-50-FPN



    Input:

        8 channels:

        red, green, blue, ndvi,

        relative_elevation, shaded_relief,

        nir, ndwi



    Classes:

        background + RTS



    Initialization:

        - COCO-pretrained Mask R-CNN when pretrained=True

        - Preserve pretrained RGB conv1 weights for channels 0:3

        - Initialize channels 3:8 using the mean of the

          pretrained RGB conv1 filters

    """



    num_classes = 2



    # --------------------------------------------------

    # Load standard Mask R-CNN

    # --------------------------------------------------



    if pretrained:

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    else:

        weights = None



    model = maskrcnn_resnet50_fpn(

        weights=weights

    )



    # --------------------------------------------------

    # Expand ResNet conv1: 3 channels -> 8 channels

    # --------------------------------------------------



    old_conv = model.backbone.body.conv1



    new_conv = nn.Conv2d(

        in_channels=8,

        out_channels=old_conv.out_channels,

        kernel_size=old_conv.kernel_size,

        stride=old_conv.stride,

        padding=old_conv.padding,

        bias=False,

    )



    with torch.no_grad():



        if pretrained:



            old_weights = old_conv.weight.data



            # Preserve pretrained RGB filters

            new_conv.weight[:, :3, :, :] = old_weights



            # Mean pretrained RGB filter

            rgb_mean = old_weights.mean(

                dim=1,

                keepdim=True,

            )



            # Initialize the five additional bands

            new_conv.weight[:, 3:, :, :] = (

                rgb_mean.repeat(1, 5, 1, 1)

            )



        else:



            # Standard Kaiming initialization when

            # no pretrained model is requested

            nn.init.kaiming_normal_(

                new_conv.weight,

                mode="fan_out",

                nonlinearity="relu",

            )



    model.backbone.body.conv1 = new_conv

    # --------------------------------------------------

    # Disable default 3-channel ImageNet normalization

    # --------------------------------------------------

    # EXP002 inputs are already standardized band-by-band

    # using statistics from the 605-image training fold.



    model.transform.image_mean = [0.0] * 8

    model.transform.image_std = [1.0] * 8

    # --------------------------------------------------

    # Replace COCO box classifier

    # --------------------------------------------------



    in_features_box = (

        model.roi_heads

        .box_predictor

        .cls_score

        .in_features

    )



    model.roi_heads.box_predictor = (

        FastRCNNPredictor(

            in_features_box,

            num_classes,

        )

    )



    # --------------------------------------------------

    # Replace COCO mask predictor

    # --------------------------------------------------



    in_features_mask = (

        model.roi_heads

        .mask_predictor

        .conv5_mask

        .in_channels

    )



    hidden_layer = 256



    model.roi_heads.mask_predictor = (

        MaskRCNNPredictor(

            in_features_mask,

            hidden_layer,

            num_classes,

        )

    )



    return model
