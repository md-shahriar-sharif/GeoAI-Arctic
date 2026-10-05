
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



PER_INSTANCE_PATH = OUTPUT_DIR / "per_instance_diagnostic.csv"

SIZE_SUMMARY_PATH = OUTPUT_DIR / "size_summary.csv"

SMALL_BINS_PATH = OUTPUT_DIR / "small_area_bins_summary.csv"



SIZE_CLASSES = ["small", "medium", "large"]

SMALL_BINS = ["10-49", "50-99", "100-199", "200-299"]

IOU_THRESHOLDS = (0.30, 0.50, 0.70)





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

    """IoU between one GT box and a set of candidate boxes."""

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



    union = gt_area + box_area - intersection

    return intersection / union.clamp(min=1e-8)





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





def model_stages(model, image, device):

    """Return RPN proposals and final boxes in original image coordinates."""

    image = image.to(device, non_blocking=True)

    original_size = (image.shape[-2], image.shape[-1])



    images, _ = model.transform([image], None)

    features = model.backbone(images.tensors)



    if isinstance(features, torch.Tensor):

        features = {"0": features}



    proposals, _ = model.rpn(images, features, None)



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



    rpn_boxes = proposals[0].clone()

    transformed_h, transformed_w = images.image_sizes[0]

    original_h, original_w = original_size



    rpn_boxes[:, [0, 2]] *= original_w / transformed_w

    rpn_boxes[:, [1, 3]] *= original_h / transformed_h



    return (

        rpn_boxes,

        detections[0]["boxes"],

        detections[0]["scores"],

    )





def best_match(gt_box, boxes, scores=None):

    ious = box_iou(gt_box, boxes)



    if ious.numel() == 0:

        return 0.0, 0.0



    index = int(torch.argmax(ious))

    best_iou = float(ious[index])



    if scores is None:

        return best_iou, 0.0



    return best_iou, float(scores[index])





def diagnose_instance(

    image_id,

    gt_index,

    gt_box,

    gt_area,

    rpn_boxes,

    final_boxes,

    final_scores,

):

    rpn_iou, _ = best_match(gt_box, rpn_boxes)

    final_iou, final_score = best_match(

        gt_box,

        final_boxes,

        final_scores,

    )



    row = {

        "image_id": image_id,

        "gt_index": gt_index,

        "gt_area": gt_area,

        "size_class": size_class(gt_area),

        "small_area_bin": small_area_bin(gt_area),

        "gt_x1": float(gt_box[0]),

        "gt_y1": float(gt_box[1]),

        "gt_x2": float(gt_box[2]),

        "gt_y2": float(gt_box[3]),

        "num_rpn_proposals": len(rpn_boxes),

        "best_rpn_box_iou": rpn_iou,

        "num_final_detections": len(final_boxes),

        "best_final_box_iou": final_iou,

        "best_final_score": final_score,

    }



    for threshold in IOU_THRESHOLDS:

        suffix = int(threshold * 100)

        row[f"rpn_recall_iou{suffix}"] = int(rpn_iou >= threshold)

        row[f"final_recall_iou{suffix}"] = int(final_iou >= threshold)



    return row





def run_diagnostic(model, dataset, device):

    rows = []



    with torch.no_grad():

        for idx in range(len(dataset)):

            image, target = dataset[idx]

            image_id = int(target["image_id"])



            rpn_boxes, final_boxes, final_scores = model_stages(

                model,

                image,

                device,

            )



            gt_boxes = target["boxes"].to(device)

            gt_areas = target["area"].numpy()



            for gt_index, (gt_box, area) in enumerate(

                zip(gt_boxes, gt_areas)

            ):

                rows.append(

                    diagnose_instance(

                        image_id,

                        gt_index,

                        gt_box,

                        float(area),

                        rpn_boxes,

                        final_boxes,

                        final_scores,

                    )

                )



            if (idx + 1) % 25 == 0:

                print(f"Processed {idx + 1}/{len(dataset)} images")



    return pd.DataFrame(rows)





def summarize(subset, label_name, label):

    """Summarize proposal and final recall for one size group."""

    row = {

        label_name: label,

        "gt_instances": len(subset),

        "mean_gt_area": subset["gt_area"].mean(),

        "mean_best_rpn_iou": subset["best_rpn_box_iou"].mean(),

        "mean_best_final_iou": subset["best_final_box_iou"].mean(),

    }



    for threshold in (30, 50, 70):

        row[f"rpn_recall_iou{threshold}"] = (

            subset[f"rpn_recall_iou{threshold}"].mean()

        )

        row[f"final_recall_iou{threshold}"] = (

            subset[f"final_recall_iou{threshold}"].mean()

        )



    return row





def build_summaries(df):

    size_summary = pd.DataFrame(

        [

            summarize(

                df[df["size_class"] == group],

                "size_class",

                group,

            )

            for group in SIZE_CLASSES

        ]

    )



    small = df[df["size_class"] == "small"]

    small_rows = []



    for area_bin in SMALL_BINS:

        subset = small[small["small_area_bin"] == area_bin]

        if not subset.empty:

            small_rows.append(

                summarize(subset, "area_bin", area_bin)

            )



    return size_summary, pd.DataFrame(small_rows)





def main():

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA GPU is required for this diagnostic.")



    device = torch.device("cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



    dataset = build_dataset()

    model = load_model(device)



    print("EXP002 small-RTS diagnostic")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Validation images: {len(dataset)}")

    print(f"Checkpoint: {CHECKPOINT_PATH.name}")



    df = run_diagnostic(model, dataset, device)

    size_summary, small_summary = build_summaries(df)



    df.to_csv(PER_INSTANCE_PATH, index=False)

    size_summary.to_csv(SIZE_SUMMARY_PATH, index=False)

    small_summary.to_csv(SMALL_BINS_PATH, index=False)



    print("\nSummary by RTS size")

    print(

        size_summary.to_string(

            index=False,

            float_format=lambda x: f"{x:.3f}",

        )

    )



    print("\nSmall RTS — fine area bins")

    print(

        small_summary.to_string(

            index=False,

            float_format=lambda x: f"{x:.3f}",

        )

    )



    print(f"\nPer-instance results: {PER_INSTANCE_PATH}")

    print(f"Size summary: {SIZE_SUMMARY_PATH}")

    print(f"Small-area summary: {SMALL_BINS_PATH}")





if __name__ == "__main__":

    main()

