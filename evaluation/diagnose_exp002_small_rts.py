import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import RTSRawDataset
from data.preprocessing_8band import RTS8BandPreprocessor
from model.model_8band import build_8band_maskrcnn


# =========================================================
# Configuration
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

per_instance_path = output_dir / "per_instance_diagnostic.csv"
size_summary_path = output_dir / "size_summary.csv"
small_bins_path = output_dir / "small_area_bins_summary.csv"


# =========================================================
# Dataset wrapper
# Same preprocessing used by EXP002 evaluation
# =========================================================

class RTSExtendedDataset(Dataset):

    def __init__(self, raw_dataset, preprocessor):
        self.raw_dataset = raw_dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):

        image, target = self.raw_dataset[idx]

        # Raw dataset returns CHW.
        # Preprocessor expects HWC.
        image_hwc = image.numpy().transpose(1, 2, 0)

        image_8band = self.preprocessor.prepare_8band_maskrcnn(
            image_hwc
        )

        image_tensor = torch.from_numpy(
            image_8band
        ).float()

        return image_tensor, target


# =========================================================
# Box IoU helper
# =========================================================

def box_iou_one_to_many(gt_box, boxes):
    """
    gt_box: Tensor [4]
    boxes:  Tensor [N, 4]

    Returns IoU for GT box against every candidate box.
    """

    if boxes.numel() == 0:
        return torch.zeros(
            (0,),
            dtype=torch.float32,
            device=gt_box.device,
        )

    x1 = torch.maximum(gt_box[0], boxes[:, 0])
    y1 = torch.maximum(gt_box[1], boxes[:, 1])
    x2 = torch.minimum(gt_box[2], boxes[:, 2])
    y2 = torch.minimum(gt_box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    intersection = inter_w * inter_h

    gt_area = (
        (gt_box[2] - gt_box[0]).clamp(min=0)
        * (gt_box[3] - gt_box[1]).clamp(min=0)
    )

    box_areas = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    union = gt_area + box_areas - intersection

    return intersection / union.clamp(min=1e-8)


# =========================================================
# Size helpers
# =========================================================

def get_size_class(area):

    if area < 300:
        return "small"

    if area < 2000:
        return "medium"

    return "large"


def get_small_area_bin(area):

    if area < 50:
        return "10-49"

    if area < 100:
        return "50-99"

    if area < 200:
        return "100-199"

    if area < 300:
        return "200-299"

    return None


# =========================================================
# Device
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this diagnostic."
    )

device = torch.device("cuda")

print("=" * 70)
print("EXP002 SMALL RTS DIAGNOSTIC")
print("=" * 70)
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Checkpoint:", checkpoint_path)
print()


# =========================================================
# Validation dataset
# =========================================================

preprocessor = RTS8BandPreprocessor(
    preprocessing_path
)

val_raw_dataset = RTSRawDataset(
    release_root=release_root,
    split_csv=split_path,
    fold="val",
)

val_dataset = RTSExtendedDataset(
    val_raw_dataset,
    preprocessor,
)

print("Validation images:", len(val_dataset))


# =========================================================
# Load EXP002 epoch 10
# =========================================================

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
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

print("Checkpoint loaded successfully.")
print()


# =========================================================
# Diagnostic
# =========================================================

rows = []

with torch.no_grad():

    for idx in range(len(val_dataset)):

        image, target = val_dataset[idx]

        image_id = int(
            target["image_id"].item()
        )

        image_gpu = image.to(
            device,
            non_blocking=True,
        )

        # -------------------------------------------------
        # 1. Run Mask R-CNN transform
        # -------------------------------------------------

        original_image_sizes = [
            (image_gpu.shape[-2], image_gpu.shape[-1])
        ]

        images, _ = model.transform(
            [image_gpu],
            None,
        )

        # -------------------------------------------------
        # 2. Backbone + FPN
        # -------------------------------------------------

        features = model.backbone(
            images.tensors
        )

        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # -------------------------------------------------
        # 3. RPN proposals
        #
        # These are proposals BEFORE RoI classification.
        # -------------------------------------------------

        proposals, _ = model.rpn(
            images,
            features,
            None,
        )

        rpn_boxes_transformed = proposals[0]

        # -------------------------------------------------
        # 4. Final detections using same features/proposals
        # -------------------------------------------------

        detections, _ = model.roi_heads(
            features,
            proposals,
            images.image_sizes,
            None,
        )

        detections = model.transform.postprocess(
            detections,
            images.image_sizes,
            original_image_sizes,
        )

        final_boxes = detections[0]["boxes"]
        final_scores = detections[0]["scores"]

        # -------------------------------------------------
        # RPN proposals are in transformed image coordinates.
        #
        # Scale them back to original image coordinates so
        # they can be compared with original GT boxes.
        # -------------------------------------------------

        transformed_h, transformed_w = images.image_sizes[0]
        original_h, original_w = original_image_sizes[0]

        scale_x = original_w / transformed_w
        scale_y = original_h / transformed_h

        rpn_boxes = rpn_boxes_transformed.clone()

        rpn_boxes[:, [0, 2]] *= scale_x
        rpn_boxes[:, [1, 3]] *= scale_y

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        gt_boxes = target["boxes"].to(device)
        gt_areas = target["area"].cpu().numpy()

        for gt_idx in range(len(gt_boxes)):

            gt_box = gt_boxes[gt_idx]
            gt_area = float(gt_areas[gt_idx])

            size_class = get_size_class(
                gt_area
            )

            small_bin = get_small_area_bin(
                gt_area
            )

            # ---------------------------------------------
            # Best RPN proposal IoU
            # ---------------------------------------------

            rpn_ious = box_iou_one_to_many(
                gt_box,
                rpn_boxes,
            )

            if len(rpn_ious) > 0:
                best_rpn_iou = float(
                    rpn_ious.max().item()
                )
            else:
                best_rpn_iou = 0.0

            # ---------------------------------------------
            # Best final detection IoU
            # ---------------------------------------------

            final_ious = box_iou_one_to_many(
                gt_box,
                final_boxes,
            )

            if len(final_ious) > 0:

                best_final_index = int(
                    torch.argmax(final_ious).item()
                )

                best_final_iou = float(
                    final_ious[
                        best_final_index
                    ].item()
                )

                best_final_score = float(
                    final_scores[
                        best_final_index
                    ].item()
                )

            else:

                best_final_iou = 0.0
                best_final_score = 0.0

            rows.append({

                "image_id": image_id,
                "gt_index": gt_idx,

                "gt_area": gt_area,
                "size_class": size_class,
                "small_area_bin": small_bin,

                "gt_x1": float(gt_box[0].item()),
                "gt_y1": float(gt_box[1].item()),
                "gt_x2": float(gt_box[2].item()),
                "gt_y2": float(gt_box[3].item()),

                "num_rpn_proposals": int(
                    len(rpn_boxes)
                ),

                "best_rpn_box_iou":
                    best_rpn_iou,

                "rpn_recall_iou30":
                    int(best_rpn_iou >= 0.30),

                "rpn_recall_iou50":
                    int(best_rpn_iou >= 0.50),

                "rpn_recall_iou70":
                    int(best_rpn_iou >= 0.70),

                "num_final_detections": int(
                    len(final_boxes)
                ),

                "best_final_box_iou":
                    best_final_iou,

                "best_final_score":
                    best_final_score,

                "final_recall_iou30":
                    int(best_final_iou >= 0.30),

                "final_recall_iou50":
                    int(best_final_iou >= 0.50),

                "final_recall_iou70":
                    int(best_final_iou >= 0.70),

            })

        if (idx + 1) % 25 == 0:

            print(
                f"Processed {idx + 1}/"
                f"{len(val_dataset)} images"
            )


# =========================================================
# Save per-instance results
# =========================================================

df = pd.DataFrame(rows)

df.to_csv(
    per_instance_path,
    index=False,
)

print()
print("Per-instance diagnostic saved:")
print(per_instance_path)


# =========================================================
# Summary by official RTS size class
# =========================================================

summary_rows = []

for size_class in [
    "small",
    "medium",
    "large",
]:

    subset = df[
        df["size_class"] == size_class
    ]

    summary_rows.append({

        "size_class": size_class,

        "gt_instances": len(subset),

        "mean_gt_area":
            subset["gt_area"].mean(),

        "rpn_recall_iou30":
            subset[
                "rpn_recall_iou30"
            ].mean(),

        "rpn_recall_iou50":
            subset[
                "rpn_recall_iou50"
            ].mean(),

        "rpn_recall_iou70":
            subset[
                "rpn_recall_iou70"
            ].mean(),

        "final_recall_iou30":
            subset[
                "final_recall_iou30"
            ].mean(),

        "final_recall_iou50":
            subset[
                "final_recall_iou50"
            ].mean(),

        "final_recall_iou70":
            subset[
                "final_recall_iou70"
            ].mean(),

        "mean_best_rpn_iou":
            subset[
                "best_rpn_box_iou"
            ].mean(),

        "mean_best_final_iou":
            subset[
                "best_final_box_iou"
            ].mean(),

    })


size_summary = pd.DataFrame(
    summary_rows
)

size_summary.to_csv(
    size_summary_path,
    index=False,
)


# =========================================================
# Fine-grained analysis inside small RTS
# =========================================================

small_df = df[
    df["size_class"] == "small"
].copy()

small_bin_order = [
    "10-49",
    "50-99",
    "100-199",
    "200-299",
]

small_summary_rows = []

for area_bin in small_bin_order:

    subset = small_df[
        small_df["small_area_bin"] == area_bin
    ]

    if len(subset) == 0:
        continue

    small_summary_rows.append({

        "area_bin": area_bin,

        "gt_instances": len(subset),

        "mean_gt_area":
            subset["gt_area"].mean(),

        "rpn_recall_iou30":
            subset[
                "rpn_recall_iou30"
            ].mean(),

        "rpn_recall_iou50":
            subset[
                "rpn_recall_iou50"
            ].mean(),

        "rpn_recall_iou70":
            subset[
                "rpn_recall_iou70"
            ].mean(),

        "final_recall_iou30":
            subset[
                "final_recall_iou30"
            ].mean(),

        "final_recall_iou50":
            subset[
                "final_recall_iou50"
            ].mean(),

        "final_recall_iou70":
            subset[
                "final_recall_iou70"
            ].mean(),

        "mean_best_rpn_iou":
            subset[
                "best_rpn_box_iou"
            ].mean(),

        "mean_best_final_iou":
            subset[
                "best_final_box_iou"
            ].mean(),

    })


small_summary = pd.DataFrame(
    small_summary_rows
)

small_summary.to_csv(
    small_bins_path,
    index=False,
)


# =========================================================
# Terminal report
# =========================================================

print()
print("=" * 70)
print("SUMMARY BY RTS SIZE")
print("=" * 70)

print(
    size_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print()
print("=" * 70)
print("SMALL RTS — FINE AREA BINS")
print("=" * 70)

print(
    small_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print("Per-instance:")
print(per_instance_path)

print("Size summary:")
print(size_summary_path)

print("Small-area summary:")
print(small_bins_path)
