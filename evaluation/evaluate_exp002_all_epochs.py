
import json

import os

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





EXP_NAME = "EXP002_8Band_MaskRCNN_SOL"

MASK_THRESHOLD = 0.5



DATA_ROOT = Path(

    os.environ.get("GEOAI_ARCTIC_DATA", PROJECT_ROOT / "competition_release")

)

SPLIT_PATH = PROJECT_ROOT / "data" / "split_v1_groupaware.csv"

PREPROCESSING_PATH = PROJECT_ROOT / "data" / "preprocessing_v1.json"



EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / EXP_NAME

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

PREDICTION_DIR = EXPERIMENT_DIR / "predictions"

VAL_GT_PATH = EXPERIMENT_DIR / "instances_val.json"

RESULTS_PATH = EXPERIMENT_DIR / "checkpoint_evaluation.csv"





class RTS8BandDataset(Dataset):

    """Apply EXP002 preprocessing to a raw RTS dataset."""



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





def build_validation_dataset():

    raw = RTSRawDataset(DATA_ROOT, SPLIT_PATH, fold="val")

    preprocessor = RTS8BandPreprocessor(PREPROCESSING_PATH)

    return RTS8BandDataset(raw, preprocessor)





def create_validation_ground_truth():

    """Create a COCO annotation file containing validation images only."""

    split = pd.read_csv(SPLIT_PATH)

    val_ids = set(

        split.loc[split["fold"] == "val", "image_id"]

        .astype(int)

        .tolist()

    )



    full_gt = (

        DATA_ROOT

        / "train"

        / "annotations"

        / "instances_train.json"

    )



    with open(full_gt, "r") as f:

        coco = json.load(f)



    val_gt = {

        "images": [

            image

            for image in coco["images"]

            if int(image["id"]) in val_ids

        ],

        "annotations": [

            ann

            for ann in coco["annotations"]

            if int(ann["image_id"]) in val_ids

        ],

        "categories": coco["categories"],

    }



    with open(VAL_GT_PATH, "w") as f:

        json.dump(val_gt, f)



    return len(val_gt["images"]), len(val_gt["annotations"])





def load_checkpoint(epoch, device):

    path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pth"



    if not path.exists():

        raise FileNotFoundError(f"Missing checkpoint: {path}")



    checkpoint = torch.load(

        path,

        map_location="cpu",

        weights_only=False,

    )



    model = build_8band_maskrcnn(pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.to(device).eval()



    return model





def predict(model, dataset, device):

    """Run validation inference and return COCO segmentation predictions."""

    predictions = []

    start = time.time()



    with torch.no_grad():

        for idx in range(len(dataset)):

            image, target = dataset[idx]

            image_id = int(target["image_id"].item())



            output = model(

                [image.to(device, non_blocking=True)]

            )[0]



            scores = output["scores"].detach().cpu().numpy()

            masks = output["masks"].detach().cpu().numpy()



            for score, mask in zip(scores, masks[:, 0]):

                binary_mask = (mask >= MASK_THRESHOLD).astype(np.uint8)



                if not binary_mask.any():

                    continue



                rle = mask_utils.encode(

                    np.asfortranarray(binary_mask)

                )

                rle["counts"] = rle["counts"].decode("utf-8")



                predictions.append(

                    {

                        "image_id": image_id,

                        "category_id": 1,

                        "segmentation": rle,

                        "score": float(score),

                    }

                )



            if (idx + 1) % 25 == 0:

                print(f"Processed {idx + 1}/{len(dataset)}")



    return predictions, time.time() - start





def evaluate_coco(prediction_path):

    """Evaluate segmentation predictions with the challenge-style settings."""

    coco_gt = COCO(str(VAL_GT_PATH))

    coco_dt = coco_gt.loadRes(str(prediction_path))



    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")

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



    s = evaluator.stats

    return {

        "coco_ap_internal": float(s[0]),

        "ap50": float(s[1]),

        "ap75": float(s[2]),

        "ap_small": float(s[3]),

        "ap_medium": float(s[4]),

        "ap_large": float(s[5]),

        "ar1": float(s[6]),

        "ar5": float(s[7]),

        "ar10": float(s[8]),

        "ar_small": float(s[9]),

        "ar_medium": float(s[10]),

        "ar_large": float(s[11]),

    }





def main():

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA GPU required for checkpoint evaluation.")



    device = torch.device("cuda")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)



    dataset = build_validation_dataset()

    n_images, n_annotations = create_validation_ground_truth()



    training_log = pd.read_csv(EXPERIMENT_DIR / "training_log.csv")

    loss_by_epoch = dict(

        zip(

            training_log["epoch"].astype(int),

            training_log["total_loss"],

        )

    )



    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(

        f"Validation: {n_images} images, "

        f"{n_annotations} RTS instances"

    )



    results = []



    for epoch in range(1, 11):

        print(f"\nEvaluating epoch {epoch:02d}")



        model = load_checkpoint(epoch, device)

        predictions, elapsed = predict(model, dataset, device)



        prediction_path = (

            PREDICTION_DIR

            / f"val_predictions_epoch{epoch:02d}.json"

        )



        with open(prediction_path, "w") as f:

            json.dump(predictions, f)



        metrics = evaluate_coco(prediction_path)



        result = {

            "epoch": epoch,

            "train_loss": float(loss_by_epoch[epoch]),

            **metrics,

            "predicted_instances": len(predictions),

            "inference_time_seconds": elapsed,

        }

        results.append(result)



        # Save after every checkpoint so partial progress is retained.

        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)



        print(

            f"Epoch {epoch:02d} "

            f"| Loss {result['train_loss']:.4f} "

            f"| COCO AP {result['coco_ap_internal']:.4f} "

            f"| AP50 {result['ap50']:.4f} "

            f"| AP75 {result['ap75']:.4f} "

            f"| Predictions {len(predictions)}"

        )



        del model

        torch.cuda.empty_cache()



    results = pd.DataFrame(results)



    columns = [

        "epoch",

        "train_loss",

        "coco_ap_internal",

        "ap50",

        "ap75",

        "ap_small",

        "ap_medium",

        "ap_large",

    ]



    print("\nCheckpoint evaluation complete")

    print(

        results[columns].to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",

        )

    )

    print(f"\nResults saved: {RESULTS_PATH}")

    print(

        "Note: final checkpoint selection should use the official "

        "competition evaluator, not this internal COCO AP value."

    )





if __name__ == "__main__":

    main()

