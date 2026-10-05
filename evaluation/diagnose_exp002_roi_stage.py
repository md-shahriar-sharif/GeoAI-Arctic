import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data.dataset import RTSRawDataset
from data.preprocessing_8band import RTS8BandPreprocessor
from model.model_8band import build_8band_maskrcnn
EXP_NAME = "EXP002_8Band_MaskRCNN_SOL"
DATA_ROOT = Path(
    os.environ.get("GEOAI_ARCTIC_DATA", PROJECT_ROOT / "competition_release")
)
SPLIT_PATH = PROJECT_ROOT / "data" / "split_v1_groupaware.csv"
PREPROCESSING_PATH = PROJECT_ROOT / "data" / "preprocessing_v1.json"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / EXP_NAME
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "epoch_10.pth"
OUTPUT_DIR = EXPERIMENT_DIR / "small_rts_diagnostic"
OUTPUT_CSV = OUTPUT_DIR / "roi_stage_diagnostic.csv"
SUMMARY_CSV = OUTPUT_DIR / "roi_stage_summary.csv"
SMALL_BINS = ["10-49", "50-99", "100-199", "200-299"]
class RTS8BandDataset(Dataset):
    """Apply EXP002 preprocessing to the raw validation dataset."""
    def __init__(self, raw_dataset, preprocessor):
        self.raw_dataset = raw_dataset
        self.preprocessor = preprocessor
    def __len__(self):
        return len(self.raw_dataset)
    def __getitem__(self, idx):
        image, target = self.raw_dataset[idx]
        image = image.numpy().transpose(1, 2, 0)
        image = self.preprocessor.prepare_8band_maskrcnn(image)
        return torch.from_numpy(image).float(), target
def box_iou(gt_box, boxes):
    """IoU between one GT box and many candidate boxes."""
    if boxes.numel() == 0:
        return torch.empty(0, device=gt_box.device)
    top_left = torch.maximum(gt_box[:2], boxes[:, :2])
    bottom_right = torch.minimum(gt_box[2:], boxes[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    gt_wh = (gt_box[2:] - gt_box[:2]).clamp(min=0)
    gt_area = gt_wh[0] * gt_wh[1]
    box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    box_area = box_wh[:, 0] * box_wh[:, 1]
    return intersection / (
        gt_area + box_area - intersection
    ).clamp(min=1e-8)
def size_class(area):
    if area < 300:
        return "small"
    if area < 2000:
        return "medium"
    return "large"
def small_area_bin(area):
    if area < 50:
        return "10-49"
    if area < 100:
        return "50-99"
    if area < 200:
        return "100-199"
    if area < 300:
        return "200-299"
    return None
def build_dataset():
    raw = RTSRawDataset(DATA_ROOT, SPLIT_PATH, fold="val")
    preprocessor = RTS8BandPreprocessor(PREPROCESSING_PATH)
    return RTS8BandDataset(raw, preprocessor)
def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model = build_8band_maskrcnn(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
def inspect_model_stages(model, image, device):
    """
    Run one image through RPN and RoI heads while keeping intermediate
    proposal classification and box-regression outputs.
    """
    image = image.to(device)
    original_size = (image.shape[-2], image.shape[-1])
    images, _ = model.transform([image], None)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    proposals, _ = model.rpn(images, features, None)
    # Classify and refine every RPN proposal before final filtering/NMS.
    roi_features = model.roi_heads.box_roi_pool(
        features,
        proposals,
        images.image_sizes,
    )
    roi_features = model.roi_heads.box_head(roi_features)
    class_logits, box_regression = model.roi_heads.box_predictor(
        roi_features
    )
    probabilities = torch.softmax(class_logits, dim=-1)
    background_prob = probabilities[:, 0]
    rts_prob = probabilities[:, 1]
    decoded = model.roi_heads.box_coder.decode(
        box_regression,
        proposals,
    )
    decoded = decoded.reshape(decoded.shape[0], -1, 4)
    # Class 1 is RTS.
    refined_rts_boxes = decoded[:, 1]
    # Standard final detections after classification, regression and NMS.
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
    )[0]
    # Proposals/refined boxes are still in transformed coordinates.
    transformed_h, transformed_w = images.image_sizes[0]
    original_h, original_w = original_size
    sx = original_w / transformed_w
    sy = original_h / transformed_h
    proposals_original = proposals[0].clone()
    proposals_original[:, [0, 2]] *= sx
    proposals_original[:, [1, 3]] *= sy
    refined_original = refined_rts_boxes.clone()
    refined_original[:, [0, 2]] *= sx
    refined_original[:, [1, 3]] *= sy
    return {
        "proposals": proposals_original,
        "rts_prob": rts_prob,
        "background_prob": background_prob,
        "refined_boxes": refined_original,
        "final_boxes": detections["boxes"],
        "final_scores": detections["scores"],
    }
def best_final_match(gt_box, boxes, scores):
    ious = box_iou(gt_box, boxes)
    if ious.numel() == 0:
        return 0.0, 0.0
    index = int(torch.argmax(ious))
    return float(ious[index]), float(scores[index])
def diagnose_gt(gt_box, area, stage):
    proposal_ious = box_iou(gt_box, stage["proposals"])
    if proposal_ious.numel() == 0:
        return None
    # Follow the best-localized RPN proposal for this GT.
    proposal_idx = int(torch.argmax(proposal_ious))
    proposal_iou = float(proposal_ious[proposal_idx])
    rts_prob = float(stage["rts_prob"][proposal_idx])
    background_prob = float(stage["background_prob"][proposal_idx])
    refined_box = stage["refined_boxes"][proposal_idx]
    refined_iou = float(
        box_iou(gt_box, refined_box.unsqueeze(0))[0]
    )
    final_iou, final_score = best_final_match(
        gt_box,
        stage["final_boxes"],
        stage["final_scores"],
    )
    return {
        "gt_area": area,
        "size_class": size_class(area),
        "small_area_bin": small_area_bin(area),
        "best_rpn_iou": proposal_iou,
        "best_rpn_good_iou50": int(proposal_iou >= 0.50),
        "roi_rts_probability": rts_prob,
        "roi_background_probability": background_prob,
        "roi_predicts_rts": int(rts_prob > background_prob),
        "refined_box_iou": refined_iou,
        "regression_improved_iou": int(refined_iou > proposal_iou),
        "final_box_iou": final_iou,
        "final_score": final_score,
        "final_recall_iou50": int(final_iou >= 0.50),
    }
def run_diagnostic(model, dataset, device):
    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target = dataset[idx]
            image_id = int(target["image_id"])
            stage = inspect_model_stages(model, image, device)
            gt_boxes = target["boxes"].to(device)
            gt_areas = target["area"].numpy()
            for gt_index, (gt_box, area) in enumerate(
                zip(gt_boxes, gt_areas)
            ):
                result = diagnose_gt(gt_box, float(area), stage)
                if result is not None:
                    rows.append(
                        {
                            "image_id": image_id,
                            "gt_index": gt_index,
                            **result,
                        }
                    )
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images")
    return pd.DataFrame(rows)
def summarize_group(subset, label):
    return {
        "group": label,
        "good_rpn_instances": len(subset),
        "mean_rpn_iou": subset["best_rpn_iou"].mean(),
        "mean_roi_rts_prob": subset["roi_rts_probability"].mean(),
        "roi_classifies_as_rts": subset["roi_predicts_rts"].mean(),
        "mean_refined_box_iou": subset["refined_box_iou"].mean(),
        "regression_improves_iou": (
            subset["regression_improved_iou"].mean()
        ),
        "final_recall_iou50": subset["final_recall_iou50"].mean(),
    }
def build_summary(df):
    # The scientific question starts after the RPN already has IoU >= 0.50.
    good_rpn = df[df["best_rpn_good_iou50"] == 1]
    rows = []
    for size, label in [
        ("small", "all-small"),
        ("medium", "medium"),
        ("large", "large"),
    ]:
        subset = good_rpn[good_rpn["size_class"] == size]
        if not subset.empty:
            rows.append(summarize_group(subset, label))
    for area_bin in SMALL_BINS:
        subset = good_rpn[
            good_rpn["small_area_bin"] == area_bin
        ]
        if not subset.empty:
            rows.append(summarize_group(subset, area_bin))
    return pd.DataFrame(rows)
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this diagnostic.")
    device = torch.device("cuda")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    model = load_model(device)
    print("EXP002 RoI-stage diagnostic")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation images: {len(dataset)}")
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    df = run_diagnostic(model, dataset, device)
    summary = build_summary(df)
    df.to_csv(OUTPUT_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\nOnly GT instances with RPN IoU >= 0.50")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print(f"\nPer-instance results: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_CSV}")
if __name__ == "__main__":
    main()
