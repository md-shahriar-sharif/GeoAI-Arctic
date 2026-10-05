
import csv

import json

import os

import random

import sys

import time

from pathlib import Path



import numpy as np

import torch

from torch.utils.data import DataLoader, Dataset



PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))



from data.dataset import RTSRawDataset

from data.preprocessing_8band import RTS8BandPreprocessor

from model.model_8band import build_8band_maskrcnn





EXP_NAME = "EXP002_8Band_MaskRCNN_SOL"

SEED = 42



DATA_ROOT = Path(

    os.environ.get("GEOAI_ARCTIC_DATA", PROJECT_ROOT / "competition_release")

)

SPLIT_PATH = PROJECT_ROOT / "data" / "split_v1_groupaware.csv"

PREPROCESSING_PATH = PROJECT_ROOT / "data" / "preprocessing_v1.json"



EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / EXP_NAME

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

LOG_PATH = EXPERIMENT_DIR / "training_log.csv"



TRAINING = {

    "batch_size": 2,

    "epochs": 10,

    "learning_rate": 0.005,

    "momentum": 0.9,

    "weight_decay": 0.0005,

    "num_workers": 2,

}



LOSS_NAMES = [

    "loss_classifier",

    "loss_box_reg",

    "loss_mask",

    "loss_objectness",

    "loss_rpn_box_reg",

]





class RTS8BandDataset(Dataset):

    """Apply EXP002 preprocessing to the raw RTS dataset."""



    def __init__(self, raw_dataset, preprocessor):

        self.raw_dataset = raw_dataset

        self.preprocessor = preprocessor



    def __len__(self):

        return len(self.raw_dataset)



    def __getitem__(self, idx):

        image, target = self.raw_dataset[idx]



        # Raw loader returns CHW; the preprocessor works in HWC.

        image = image.numpy().transpose(1, 2, 0)

        image = self.preprocessor.prepare_8band_maskrcnn(image)



        return torch.from_numpy(image).float(), target





def collate_fn(batch):

    images, targets = zip(*batch)

    return list(images), list(targets)





def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False





def build_dataloaders():

    preprocessor = RTS8BandPreprocessor(PREPROCESSING_PATH)



    def make_dataset(fold):

        raw = RTSRawDataset(

            release_root=DATA_ROOT,

            split_csv=SPLIT_PATH,

            fold=fold,

        )

        return RTS8BandDataset(raw, preprocessor)



    train_dataset = make_dataset("train")

    val_dataset = make_dataset("val")



    train_loader = DataLoader(

        train_dataset,

        batch_size=TRAINING["batch_size"],

        shuffle=True,

        num_workers=TRAINING["num_workers"],

        collate_fn=collate_fn,

        pin_memory=True,

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=TRAINING["batch_size"],

        shuffle=False,

        num_workers=TRAINING["num_workers"],

        collate_fn=collate_fn,

        pin_memory=True,

    )



    return train_dataset, val_dataset, train_loader, val_loader





def experiment_config():

    return {

        "experiment_name": EXP_NAME,

        "model": {

            "architecture": "Mask R-CNN",

            "backbone": "ResNet-50-FPN",

            "initialization": "COCO pretrained",

            "num_classes": 2,

            "input_bands": [

                "red",

                "green",

                "blue",

                "ndvi",

                "relative_elevation",

                "shaded_relief",

                "nir",

                "ndwi",

            ],

        },

        "data": {

            "split": "split_v1_groupaware.csv",

            "train_images": 605,

            "val_images": 151,

            "nan_policy": "training-band mean imputation",

            "normalization": "training-fold per-band z-score",

            "input_channels": 8,

        },

        "training": {

            "batch_size": TRAINING["batch_size"],

            "epochs": TRAINING["epochs"],

            "optimizer": "SGD",

            "learning_rate": TRAINING["learning_rate"],

            "momentum": TRAINING["momentum"],

            "weight_decay": TRAINING["weight_decay"],

            "seed": SEED,

            "num_workers": TRAINING["num_workers"],

        },

        "hardware": {

            "platform": "ASU Sol",

            "gpu": (

                torch.cuda.get_device_name(0)

                if torch.cuda.is_available()

                else "CPU"

            ),

        },

    }





def move_batch_to_device(images, targets, device):

    images = [image.to(device, non_blocking=True) for image in images]



    targets = [

        {

            key: value.to(device, non_blocking=True)

            if torch.is_tensor(value)

            else value

            for key, value in target.items()

        }

        for target in targets

    ]



    return images, targets





def train_one_epoch(model, loader, optimizer, device, epoch):

    model.train()

    totals = {name: 0.0 for name in LOSS_NAMES}

    totals["total_loss"] = 0.0



    start = time.time()



    for batch_idx, (images, targets) in enumerate(loader, start=1):

        images, targets = move_batch_to_device(images, targets, device)



        optimizer.zero_grad(set_to_none=True)

        loss_dict = model(images, targets)

        loss = sum(loss_dict.values())



        if not torch.isfinite(loss):

            raise RuntimeError(

                f"Non-finite loss at epoch {epoch}, "

                f"batch {batch_idx}: {loss.item()}"

            )



        loss.backward()

        optimizer.step()



        for name in LOSS_NAMES:

            totals[name] += loss_dict[name].item()

        totals["total_loss"] += loss.item()



        if batch_idx % 50 == 0:

            print(

                f"Epoch {epoch}/{TRAINING['epochs']} "

                f"| Batch {batch_idx}/{len(loader)} "

                f"| Loss {loss.item():.4f}",

                flush=True,

            )



    averages = {

        name: value / len(loader)

        for name, value in totals.items()

    }



    return averages, time.time() - start





def initialize_log():

    columns = ["epoch", "total_loss", *LOSS_NAMES, "epoch_time_seconds"]



    with open(LOG_PATH, "w", newline="") as f:

        csv.writer(f).writerow(columns)





def append_log(epoch, losses, elapsed):

    row = [

        epoch,

        losses["total_loss"],

        *[losses[name] for name in LOSS_NAMES],

        elapsed,

    ]



    with open(LOG_PATH, "a", newline="") as f:

        csv.writer(f).writerow(row)





def save_checkpoint(model, optimizer, epoch, losses, config):

    path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pth"



    torch.save(

        {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "loss": losses["total_loss"],

            "seed": SEED,

            "experiment": EXP_NAME,

            "config": config,

        },

        path,

    )



    return path





def main():

    set_seed()



    if not torch.cuda.is_available():

        raise RuntimeError("EXP002 is intended to run on a CUDA GPU.")



    device = torch.device("cuda")



    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)



    config = experiment_config()

    with open(EXPERIMENT_DIR / "config.json", "w") as f:

        json.dump(config, f, indent=2)



    train_dataset, val_dataset, train_loader, _ = build_dataloaders()



    model = build_8band_maskrcnn(pretrained=True).to(device)

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=TRAINING["learning_rate"],

        momentum=TRAINING["momentum"],

        weight_decay=TRAINING["weight_decay"],

    )



    initialize_log()



    print(f"Starting {EXP_NAME}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(

        f"Train: {len(train_dataset)} | "

        f"Validation: {len(val_dataset)} | "

        f"Batch: {TRAINING['batch_size']} | "

        f"Epochs: {TRAINING['epochs']}"

    )



    for epoch in range(1, TRAINING["epochs"] + 1):

        losses, elapsed = train_one_epoch(

            model,

            train_loader,

            optimizer,

            device,

            epoch,

        )



        append_log(epoch, losses, elapsed)

        checkpoint = save_checkpoint(

            model,

            optimizer,

            epoch,

            losses,

            config,

        )



        print(

            f"Epoch {epoch}/{TRAINING['epochs']} "

            f"| Loss {losses['total_loss']:.4f} "

            f"| {elapsed:.1f}s "

            f"| Saved {checkpoint.name}",

            flush=True,

        )



    print(f"Training complete. Log: {LOG_PATH}")





if __name__ == "__main__":

    main()

