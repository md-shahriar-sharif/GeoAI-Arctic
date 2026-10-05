import os
import sys

from pathlib import Path



import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset

from torchvision.models.detection.roi_heads import fastrcnn_loss





PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))



from data.dataset import RTSRawDataset

from data.preprocessing_8band import RTS8BandPreprocessor

from model.model_8band import build_8band_maskrcnn





# =========================================================

# Paths

# =========================================================



release_root = Path(

    os.environ.get(

        "GEOAI_ARCTIC_DATA",

        PROJECT_ROOT / "competition_release",

    )

)

split_path = PROJECT_ROOT / "data" / "split_v1_groupaware.csv"

preprocessing_path = PROJECT_ROOT / "data" / "preprocessing_v1.json"



checkpoint_path = (

    PROJECT_ROOT

    / "experiments"

    / "EXP002_8Band_MaskRCNN_SOL"

    / "checkpoints"

    / "epoch_10.pth"

)



output_dir = (

    PROJECT_ROOT

    / "experiments"

    / "EXP002_8Band_MaskRCNN_SOL"

    / "small_rts_diagnostic"

)



output_dir.mkdir(parents=True, exist_ok=True)



output_csv = output_dir / "roi_stage_diagnostic.csv"

summary_csv = output_dir / "roi_stage_summary.csv"





# =========================================================

# Dataset

# =========================================================



class RTSExtendedDataset(Dataset):



    def __init__(self, raw_dataset, preprocessor):

        self.raw_dataset = raw_dataset

        self.preprocessor = preprocessor



    def __len__(self):

        return len(self.raw_dataset)



    def __getitem__(self, idx):



        image, target = self.raw_dataset[idx]



        image_hwc = image.numpy().transpose(1, 2, 0)



        image_8band = self.preprocessor.prepare_8band_maskrcnn(

            image_hwc

        )



        return (

            torch.from_numpy(image_8band).float(),

            target,

        )





# =========================================================

# IoU

# =========================================================



def box_iou_one_to_many(gt_box, boxes):



    if boxes.numel() == 0:

        return torch.zeros(

            0,

            device=gt_box.device,

        )



    x1 = torch.maximum(gt_box[0], boxes[:, 0])

    y1 = torch.maximum(gt_box[1], boxes[:, 1])

    x2 = torch.minimum(gt_box[2], boxes[:, 2])

    y2 = torch.minimum(gt_box[3], boxes[:, 3])



    iw = (x2 - x1).clamp(min=0)

    ih = (y2 - y1).clamp(min=0)



    inter = iw * ih



    gt_area = (

        (gt_box[2] - gt_box[0]).clamp(min=0)

        * (gt_box[3] - gt_box[1]).clamp(min=0)

    )



    box_area = (

        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)

        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    )



    union = gt_area + box_area - inter



    return inter / union.clamp(min=1e-8)





def size_class(area):



    if area < 300:

        return "small"

    elif area < 2000:

        return "medium"

    else:

        return "large"





def small_bin(area):



    if area < 50:

        return "10-49"

    elif area < 100:

        return "50-99"

    elif area < 200:

        return "100-199"

    elif area < 300:

        return "200-299"

    return None





# =========================================================

# Device / model

# =========================================================



if not torch.cuda.is_available():

    raise RuntimeError("CUDA GPU required.")



device = torch.device("cuda")



print("=" * 70)

print("EXP002 ROI-STAGE DIAGNOSTIC")

print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))





preprocessor = RTS8BandPreprocessor(

    preprocessing_path

)



raw_dataset = RTSRawDataset(

    release_root=release_root,

    split_csv=split_path,

    fold="val",

)



dataset = RTSExtendedDataset(

    raw_dataset,

    preprocessor,

)





model = build_8band_maskrcnn(

    pretrained=False

)



checkpoint = torch.load(

    checkpoint_path,

    map_location="cpu",

    weights_only=False,

)



model.load_state_dict(

    checkpoint["model_state_dict"],

    strict=True,

)



model.to(device)

model.eval()



print("Validation images:", len(dataset))

print("Checkpoint loaded.")

print()





# =========================================================

# Diagnostic

# =========================================================



rows = []



with torch.no_grad():



    for idx in range(len(dataset)):



        image, target = dataset[idx]



        image_id = int(

            target["image_id"].item()

        )



        image_gpu = image.to(device)



        original_size = (

            image_gpu.shape[-2],

            image_gpu.shape[-1],

        )



        # ---------------------------------------------

        # Transform

        # ---------------------------------------------



        images, _ = model.transform(

            [image_gpu],

            None,

        )



        # ---------------------------------------------

        # Backbone/FPN

        # ---------------------------------------------



        features = model.backbone(

            images.tensors

        )



        if isinstance(features, torch.Tensor):

            features = {"0": features}



        # ---------------------------------------------

        # RPN

        # ---------------------------------------------



        proposals, _ = model.rpn(

            images,

            features,

            None,

        )



        proposal_boxes = proposals[0]



        # ---------------------------------------------

        # Extract RoI features for EVERY RPN proposal

        # before final classification/NMS.

        # ---------------------------------------------



        box_features = model.roi_heads.box_roi_pool(

            features,

            proposals,

            images.image_sizes,

        )



        box_features = model.roi_heads.box_head(

            box_features

        )



        class_logits, box_regression = (

            model.roi_heads.box_predictor(

                box_features

            )

        )



        probabilities = torch.softmax(

            class_logits,

            dim=-1,

        )



        # category 0 = background

        # category 1 = RTS

        rts_probabilities = probabilities[:, 1]

        background_probabilities = probabilities[:, 0]



        # ---------------------------------------------

        # Decode box regression for every proposal

        # ---------------------------------------------



        pred_boxes = (

            model.roi_heads.box_coder.decode(

                box_regression,

                proposals,

            )

        )



        # Shape is [N, num_classes, 4]

        pred_boxes = pred_boxes.reshape(

            pred_boxes.shape[0],

            -1,

            4,

        )



        # RTS class = index 1

        refined_rts_boxes = pred_boxes[:, 1, :]



        # ---------------------------------------------

        # Final normal detections

        # ---------------------------------------------



        detections, _ = model.roi_heads(

            features,

            proposals,

            images.image_sizes,

            None,

        )



        detections = model.transform.postprocess(

            detections,

            images.image_sizes,

            [original_size],

        )



        final_boxes = detections[0]["boxes"]

        final_scores = detections[0]["scores"]



        # ---------------------------------------------

        # Convert transformed coordinates back to

        # original image coordinates

        # ---------------------------------------------



        transformed_h, transformed_w = (

            images.image_sizes[0]

        )



        original_h, original_w = original_size



        sx = original_w / transformed_w

        sy = original_h / transformed_h



        proposals_original = proposal_boxes.clone()

        proposals_original[:, [0, 2]] *= sx

        proposals_original[:, [1, 3]] *= sy



        refined_original = refined_rts_boxes.clone()

        refined_original[:, [0, 2]] *= sx

        refined_original[:, [1, 3]] *= sy



        # ---------------------------------------------

        # Ground truth

        # ---------------------------------------------



        gt_boxes = target["boxes"].to(device)

        gt_areas = target["area"].cpu().numpy()



        for gt_idx in range(len(gt_boxes)):



            gt_box = gt_boxes[gt_idx]

            area = float(gt_areas[gt_idx])



            proposal_ious = box_iou_one_to_many(

                gt_box,

                proposals_original,

            )



            if len(proposal_ious) == 0:

                continue



            # Best RPN proposal for this GT

            best_prop_idx = int(

                torch.argmax(proposal_ious).item()

            )



            best_prop_iou = float(

                proposal_ious[

                    best_prop_idx

                ].item()

            )



            # What did classifier think about THAT

            # best-localized proposal?

            best_prop_rts_prob = float(

                rts_probabilities[

                    best_prop_idx

                ].item()

            )



            best_prop_bg_prob = float(

                background_probabilities[

                    best_prop_idx

                ].item()

            )



            # What did box regression do to THAT proposal?

            refined_box = refined_original[

                best_prop_idx

            ]



            refined_iou = float(

                box_iou_one_to_many(

                    gt_box,

                    refined_box.unsqueeze(0),

                )[0].item()

            )



            # Best final output

            final_ious = box_iou_one_to_many(

                gt_box,

                final_boxes,

            )



            if len(final_ious) > 0:



                best_final_idx = int(

                    torch.argmax(final_ious).item()

                )



                best_final_iou = float(

                    final_ious[

                        best_final_idx

                    ].item()

                )



                best_final_score = float(

                    final_scores[

                        best_final_idx

                    ].item()

                )



            else:



                best_final_iou = 0.0

                best_final_score = 0.0



            rows.append({



                "image_id": image_id,

                "gt_index": gt_idx,

                "gt_area": area,



                "size_class":

                    size_class(area),



                "small_area_bin":

                    small_bin(area),



                "best_rpn_iou":

                    best_prop_iou,



                "best_rpn_good_iou50":

                    int(best_prop_iou >= 0.50),



                "roi_rts_probability":

                    best_prop_rts_prob,



                "roi_background_probability":

                    best_prop_bg_prob,



                "roi_predicts_rts":

                    int(

                        best_prop_rts_prob

                        >

                        best_prop_bg_prob

                    ),



                "refined_box_iou":

                    refined_iou,



                "regression_improved_iou":

                    int(

                        refined_iou

                        >

                        best_prop_iou

                    ),



                "final_box_iou":

                    best_final_iou,



                "final_score":

                    best_final_score,



                "final_recall_iou50":

                    int(

                        best_final_iou >= 0.50

                    ),



            })



        if (idx + 1) % 25 == 0:

            print(

                f"Processed {idx + 1}/"

                f"{len(dataset)} images"

            )





# =========================================================

# Save

# =========================================================



df = pd.DataFrame(rows)



df.to_csv(

    output_csv,

    index=False,

)





# =========================================================

# Focus specifically on GTs where RPN had a GOOD proposal

# =========================================================



good_rpn = df[

    df["best_rpn_good_iou50"] == 1

].copy()





summary_rows = []



groups = [

    ("small", "all-small"),

    ("medium", "medium"),

    ("large", "large"),

]





for size, label in groups:



    subset = good_rpn[

        good_rpn["size_class"] == size

    ]



    if len(subset) == 0:

        continue



    summary_rows.append({



        "group": label,



        "good_rpn_instances":

            len(subset),



        "mean_rpn_iou":

            subset[

                "best_rpn_iou"

            ].mean(),



        "mean_roi_rts_prob":

            subset[

                "roi_rts_probability"

            ].mean(),



        "roi_classifies_as_rts":

            subset[

                "roi_predicts_rts"

            ].mean(),



        "mean_refined_box_iou":

            subset[

                "refined_box_iou"

            ].mean(),



        "regression_improves_iou":

            subset[

                "regression_improved_iou"

            ].mean(),



        "final_recall_iou50":

            subset[

                "final_recall_iou50"

            ].mean(),



    })





# Tiny/small subgroups

for area_bin in [

    "10-49",

    "50-99",

    "100-199",

    "200-299",

]:



    subset = good_rpn[

        good_rpn["small_area_bin"]

        == area_bin

    ]



    if len(subset) == 0:

        continue



    summary_rows.append({



        "group": area_bin,



        "good_rpn_instances":

            len(subset),



        "mean_rpn_iou":

            subset[

                "best_rpn_iou"

            ].mean(),



        "mean_roi_rts_prob":

            subset[

                "roi_rts_probability"

            ].mean(),



        "roi_classifies_as_rts":

            subset[

                "roi_predicts_rts"

            ].mean(),



        "mean_refined_box_iou":

            subset[

                "refined_box_iou"

            ].mean(),



        "regression_improves_iou":

            subset[

                "regression_improved_iou"

            ].mean(),



        "final_recall_iou50":

            subset[

                "final_recall_iou50"

            ].mean(),



    })





summary = pd.DataFrame(

    summary_rows

)



summary.to_csv(

    summary_csv,

    index=False,

)





print()

print("=" * 85)

print("ROI-STAGE DIAGNOSTIC")

print("ONLY GT INSTANCES WITH RPN IoU >= 0.50")

print("=" * 85)



print(

    summary.to_string(

        index=False,

        float_format=lambda x: f"{x:.3f}",

    )

)



print()

print("Per-instance results:")

print(output_csv)



print()

print("Summary:")

print(summary_csv)



print()

print("=" * 85)

print("DONE")

print("=" * 85)
