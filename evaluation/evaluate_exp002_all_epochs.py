import os
import csv

import json

import sys

import time

from pathlib import Path



import numpy as np

import pandas as pd

import torch

from pycocotools import mask as mask_utils

from pycocotools.coco import COCO

from pycocotools.cocoeval import COCOeval

from torch.utils.data import Dataset



PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))



from data.dataset import RTSRawDataset

from data.preprocessing_8band import RTS8BandPreprocessor

from model.model_8band import build_8band_maskrcnn





# ---------------------------------------------------------

# Paths

# ---------------------------------------------------------



EXP_NAME = "EXP002_8Band_MaskRCNN_SOL"



release_root = Path(

    os.environ.get(

        "GEOAI_ARCTIC_DATA",

        PROJECT_ROOT / "competition_release",

    )

)

split_path = PROJECT_ROOT / "data" / "split_v1_groupaware.csv"

preprocessing_path = PROJECT_ROOT / "data" / "preprocessing_v1.json"



experiment_dir = PROJECT_ROOT / "experiments" / EXP_NAME

checkpoint_dir = experiment_dir / "checkpoints"

prediction_dir = experiment_dir / "predictions"



val_gt_path = experiment_dir / "instances_val.json"

results_path = experiment_dir / "checkpoint_evaluation.csv"



prediction_dir.mkdir(parents=True, exist_ok=True)



MASK_THRESHOLD = 0.5





# ---------------------------------------------------------

# Dataset wrapper: EXP002 8-band preprocessing

# ---------------------------------------------------------



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



        image_tensor = torch.from_numpy(image_8band).float()



        return image_tensor, target





# ---------------------------------------------------------

# Device

# ---------------------------------------------------------



if not torch.cuda.is_available():

    raise RuntimeError(

        "CUDA GPU required for EXP002 checkpoint evaluation."

    )



device = torch.device("cuda")



print("Device:", device)

print("GPU:", torch.cuda.get_device_name(0))





# ---------------------------------------------------------

# Validation dataset

# ---------------------------------------------------------



preprocessor = RTS8BandPreprocessor(

    preprocessing_path

)



val_raw_dataset = RTSRawDataset(

    release_root=release_root,

    split_csv=split_path,

    fold="val",

)



band8_val_dataset = RTSExtendedDataset(

    val_raw_dataset,

    preprocessor,

)



print("Validation images:", len(band8_val_dataset))





# ---------------------------------------------------------

# Create validation-only COCO ground truth

# ---------------------------------------------------------



split_df = pd.read_csv(split_path)



val_image_ids = set(

    split_df.loc[

        split_df["fold"] == "val",

        "image_id"

    ].astype(int).tolist()

)



full_gt_path = (

    release_root

    / "train"

    / "annotations"

    / "instances_train.json"

)



with open(full_gt_path, "r") as f:

    coco_full = json.load(f)



val_images = [

    img

    for img in coco_full["images"]

    if int(img["id"]) in val_image_ids

]



val_annotations = [

    ann

    for ann in coco_full["annotations"]

    if int(ann["image_id"]) in val_image_ids

]



coco_val_gt = {

    "images": val_images,

    "annotations": val_annotations,

    "categories": coco_full["categories"],

}



with open(val_gt_path, "w") as f:

    json.dump(coco_val_gt, f)



print("Validation GT images:", len(val_images))

print("Validation GT annotations:", len(val_annotations))





# ---------------------------------------------------------

# Evaluation helper

# ---------------------------------------------------------



def evaluate_predictions(prediction_path):



    coco_gt = COCO(str(val_gt_path))

    coco_dt = coco_gt.loadRes(str(prediction_path))



    evaluator = COCOeval(

        coco_gt,

        coco_dt,

        iouType="segm",

    )



    # Match competition evaluator

    evaluator.params.maxDets = [1, 5, 10]



    evaluator.params.areaRng = [

        [0, 1e10],

        [0, 300],

        [300, 2000],

        [2000, 1e10],

    ]



    evaluator.params.areaRngLbl = [

        "all",

        "small",

        "medium",

        "large",

    ]



    evaluator.evaluate()

    evaluator.accumulate()

    evaluator.summarize()



    stats = evaluator.stats



    return {

        "mask_ap_internal_invalid": float(stats[0]),

        "ap50": float(stats[1]),

        "ap75": float(stats[2]),

        "ap_small": float(stats[3]),

        "ap_medium": float(stats[4]),

        "ap_large": float(stats[5]),

        "ar1": float(stats[6]),

        "ar5": float(stats[7]),

        "ar10": float(stats[8]),

        "ar_small": float(stats[9]),

        "ar_medium": float(stats[10]),

        "ar_large": float(stats[11]),

    }





# ---------------------------------------------------------

# Load training losses

# ---------------------------------------------------------



training_log_path = (

    experiment_dir / "training_log.csv"

)



training_log = pd.read_csv(

    training_log_path

)



loss_by_epoch = dict(

    zip(

        training_log["epoch"].astype(int),

        training_log["total_loss"],

    )

)





# ---------------------------------------------------------

# Evaluate every checkpoint

# ---------------------------------------------------------



results = []



for epoch in range(1, 11):



    print()

    print("=" * 70)

    print(f"Evaluating epoch {epoch:02d}")

    print("=" * 70)



    checkpoint_path = (

        checkpoint_dir

        / f"epoch_{epoch:02d}.pth"

    )



    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Missing checkpoint: {checkpoint_path}"

        )



    # -----------------------------------------------------

    # Recreate model

    # -----------------------------------------------------



    # No COCO download is needed here because checkpoint

    # contains the complete trained state dictionary.

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



    # -----------------------------------------------------

    # Validation inference

    # -----------------------------------------------------



    coco_predictions = []



    start_time = time.time()



    with torch.no_grad():



        for idx in range(len(band8_val_dataset)):



            image, target = band8_val_dataset[idx]



            image_id = int(

                target["image_id"].item()

            )



            image_gpu = image.to(

                device,

                non_blocking=True,

            )



            output = model([image_gpu])[0]



            scores = (

                output["scores"]

                .detach()

                .cpu()

                .numpy()

            )



            masks = (

                output["masks"]

                .detach()

                .cpu()

                .numpy()

            )



            for i in range(len(scores)):



                mask_probability = masks[i, 0]



                binary_mask = (

                    mask_probability >= MASK_THRESHOLD

                ).astype(np.uint8)



                if binary_mask.sum() == 0:

                    continue



                rle = mask_utils.encode(

                    np.asfortranarray(binary_mask)

                )



                rle["counts"] = (

                    rle["counts"].decode("utf-8")

                )



                coco_predictions.append({

                    "image_id": image_id,

                    "category_id": 1,

                    "segmentation": rle,

                    "score": float(scores[i]),

                })



            if (idx + 1) % 25 == 0:

                print(

                    f"Processed {idx + 1}/"

                    f"{len(band8_val_dataset)}"

                )



    inference_time = time.time() - start_time



    # -----------------------------------------------------

    # Save prediction JSON

    # -----------------------------------------------------



    prediction_path = (

        prediction_dir

        / f"val_predictions_epoch{epoch:02d}.json"

    )



    with open(prediction_path, "w") as f:

        json.dump(coco_predictions, f)



    print(

        "Predicted instances:",

        len(coco_predictions),

    )



    print(

        f"Inference time: {inference_time:.1f}s"

    )



    # -----------------------------------------------------

    # COCO evaluation

    # -----------------------------------------------------



    metrics = evaluate_predictions(

        prediction_path

    )



    result = {

        "epoch": epoch,

        "train_loss": float(

            loss_by_epoch[epoch]

        ),

        **metrics,

        "predicted_instances": len(

            coco_predictions

        ),

        "inference_time_seconds": (

            inference_time

        ),

    }



    results.append(result)



    # Save after every epoch so partial progress survives

    pd.DataFrame(results).to_csv(

        results_path,

        index=False,

    )



    print(

        f"EPOCH {epoch:02d} "

        f"| Train loss: "

        f"{result['train_loss']:.4f} "

        f"| Mask AP: "

        f"{result['mask_ap_internal_invalid']:.4f} "

        f"| AP50: "

        f"{result['ap50']:.4f} "

        f"| AP75: "

        f"{result['ap75']:.4f}"

    )



    # Release model memory before next checkpoint

    del model

    del checkpoint



    torch.cuda.empty_cache()





# ---------------------------------------------------------

# Final summary

# ---------------------------------------------------------



results_df = pd.DataFrame(results)



# Best checkpoint will be selected using the official competition evaluator.

# Internal COCOeval stats[0] is not used for checkpoint selection.



print()

print("=" * 70)

print("EXP002 CHECKPOINT EVALUATION COMPLETE")

print("=" * 70)



print(

    results_df[

        [

            "epoch",

            "train_loss",

            "mask_ap",

            "ap50",

            "ap75",

            "ap_small",

            "ap_medium",

            "ap_large",

        ]

    ].to_string(

        index=False,

        float_format=lambda x: f"{x:.4f}",

    )

)



print()

print(

    "BEST 8-BAND CHECKPOINT:"

    f" epoch_{int(best['epoch']):02d}.pth"

)



print(

    f"Best Mask AP: {best['mask_ap']:.4f}"

)



print(

    "Results saved:",

    results_path,

)
