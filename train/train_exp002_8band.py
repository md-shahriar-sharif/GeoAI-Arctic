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



# ---------------------------------------------------------

# Project setup

# ---------------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))



from data.dataset import RTSRawDataset

from data.preprocessing_8band import RTS8BandPreprocessor

from model.model_8band import build_8band_maskrcnn





# ---------------------------------------------------------

# Experiment

# ---------------------------------------------------------



EXP_NAME = "EXP002_8Band_MaskRCNN_SOL"

SEED = 42



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



experiment_dir.mkdir(parents=True, exist_ok=True)

checkpoint_dir.mkdir(parents=True, exist_ok=True)

prediction_dir.mkdir(parents=True, exist_ok=True)





# ---------------------------------------------------------

# Reproducibility

# ---------------------------------------------------------



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)



torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False





# ---------------------------------------------------------

# Configuration

# ---------------------------------------------------------



exp_config = {

    "experiment_name": EXP_NAME,



    "model": {

        "architecture": "Mask R-CNN",

        "backbone": "ResNet-50-FPN",

        "initialization": "COCO pretrained",

        "num_classes": 2,

        "input_bands": ["red", "green", "blue", "ndvi", "relative_elevation", "shaded_relief", "nir", "ndwi"],

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

        "batch_size": 2,

        "epochs": 10,

        "optimizer": "SGD",

        "learning_rate": 0.005,

        "momentum": 0.9,

        "weight_decay": 0.0005,

        "seed": SEED,

        "num_workers": 2,

    },



    "hardware": {

        "platform": "ASU Sol",

        "gpu": torch.cuda.get_device_name(0)

        if torch.cuda.is_available()

        else "CPU",

    },

}



config_path = experiment_dir / "config.json"



with open(config_path, "w") as f:

    json.dump(exp_config, f, indent=2)





# ---------------------------------------------------------

# 8-band dataset wrapper

# ---------------------------------------------------------



class RTSExtendedDataset(Dataset):

    """

    Combine RTSRawDataset with EXP002 8-band preprocessing.

    """



    def __init__(self, raw_dataset, preprocessor):

        self.raw_dataset = raw_dataset

        self.preprocessor = preprocessor



    def __len__(self):

        return len(self.raw_dataset)



    def __getitem__(self, idx):

        image, target = self.raw_dataset[idx]



        # Raw loader returns CHW. Preprocessor expects HWC.

        image_hwc = image.numpy().transpose(1, 2, 0)



        image_8band = self.preprocessor.prepare_8band_maskrcnn(image_hwc)



        image_tensor = torch.from_numpy(image_8band).float()



        return image_tensor, target





def detection_collate_fn(batch):

    images, targets = zip(*batch)

    return list(images), list(targets)





# ---------------------------------------------------------

# Datasets

# ---------------------------------------------------------



preprocessor = RTS8BandPreprocessor(preprocessing_path)



train_raw_dataset = RTSRawDataset(

    release_root=release_root,

    split_csv=split_path,

    fold="train",

)



val_raw_dataset = RTSRawDataset(

    release_root=release_root,

    split_csv=split_path,

    fold="val",

)



band8_train_dataset = RTSExtendedDataset(

    train_raw_dataset,

    preprocessor,

)



band8_val_dataset = RTSExtendedDataset(

    val_raw_dataset,

    preprocessor,

)





# ---------------------------------------------------------

# DataLoaders

# ---------------------------------------------------------



train_loader = DataLoader(

    band8_train_dataset,

    batch_size=exp_config["training"]["batch_size"],

    shuffle=True,

    num_workers=exp_config["training"]["num_workers"],

    collate_fn=detection_collate_fn,

    pin_memory=True,

)



val_loader = DataLoader(

    band8_val_dataset,

    batch_size=exp_config["training"]["batch_size"],

    shuffle=False,

    num_workers=exp_config["training"]["num_workers"],

    collate_fn=detection_collate_fn,

    pin_memory=True,

)





# ---------------------------------------------------------

# Device

# ---------------------------------------------------------



device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)



if device.type != "cuda":

    raise RuntimeError(

        "EXP002_SOL is intended to run on a CUDA GPU."

    )





# ---------------------------------------------------------

# Model

# ---------------------------------------------------------



model = build_8band_maskrcnn(

    pretrained=True

)



model = model.to(device)





# ---------------------------------------------------------

# Optimizer

# ---------------------------------------------------------



optimizer = torch.optim.SGD(

    model.parameters(),

    lr=exp_config["training"]["learning_rate"],

    momentum=exp_config["training"]["momentum"],

    weight_decay=exp_config["training"]["weight_decay"],

)



num_epochs = exp_config["training"]["epochs"]





# ---------------------------------------------------------

# Training log

# ---------------------------------------------------------



log_path = experiment_dir / "training_log.csv"



with open(log_path, "w", newline="") as f:

    writer = csv.writer(f)



    writer.writerow([

        "epoch",

        "total_loss",

        "loss_classifier",

        "loss_box_reg",

        "loss_mask",

        "loss_objectness",

        "loss_rpn_box_reg",

        "epoch_time_seconds",

    ])





# ---------------------------------------------------------

# Experiment summary

# ---------------------------------------------------------



print("=" * 70)

print("Starting:", EXP_NAME)

print("=" * 70)



print("Device:", device)

print("GPU:", torch.cuda.get_device_name(0))

print("PyTorch:", torch.__version__)

print("CUDA runtime:", torch.version.cuda)



print()



print("Training images:", len(band8_train_dataset))

print("Validation images:", len(band8_val_dataset))

print("Batch size:", exp_config["training"]["batch_size"])

print("Epochs:", num_epochs)

print("Seed:", SEED)



print()



print("Experiment directory:", experiment_dir)

print("Checkpoint directory:", checkpoint_dir)

print("Training log:", log_path)



print("=" * 70)





# ---------------------------------------------------------

# Training

# ---------------------------------------------------------



for epoch in range(1, num_epochs + 1):



    model.train()



    epoch_start = time.time()



    running_losses = {

        "loss_classifier": 0.0,

        "loss_box_reg": 0.0,

        "loss_mask": 0.0,

        "loss_objectness": 0.0,

        "loss_rpn_box_reg": 0.0,

        "total_loss": 0.0,

    }



    num_batches = 0



    for batch_idx, (images, targets) in enumerate(

        train_loader,

        start=1,

    ):



        images = [

            image.to(

                device,

                non_blocking=True,

            )

            for image in images

        ]



        targets = [

            {

                key: (

                    value.to(

                        device,

                        non_blocking=True,

                    )

                    if torch.is_tensor(value)

                    else value

                )

                for key, value in target.items()

            }

            for target in targets

        ]



        optimizer.zero_grad(

            set_to_none=True

        )



        # Forward pass

        loss_dict = model(

            images,

            targets,

        )



        loss = sum(

            loss_dict.values()

        )



        # Safety check

        if not torch.isfinite(loss):

            raise RuntimeError(

                "Non-finite loss detected "

                f"at epoch {epoch}, "

                f"batch {batch_idx}: "

                f"{loss.item()}"

            )



        # Backpropagation

        loss.backward()



        optimizer.step()



        # Record losses

        running_losses["loss_classifier"] += (

            loss_dict["loss_classifier"].item()

        )



        running_losses["loss_box_reg"] += (

            loss_dict["loss_box_reg"].item()

        )



        running_losses["loss_mask"] += (

            loss_dict["loss_mask"].item()

        )



        running_losses["loss_objectness"] += (

            loss_dict["loss_objectness"].item()

        )



        running_losses["loss_rpn_box_reg"] += (

            loss_dict["loss_rpn_box_reg"].item()

        )



        running_losses["total_loss"] += (

            loss.item()

        )



        num_batches += 1



        if batch_idx % 50 == 0:

            print(

                f"Epoch {epoch}/{num_epochs} "

                f"| Batch {batch_idx}/{len(train_loader)} "

                f"| Loss {loss.item():.4f}",

                flush=True,

            )





    # -----------------------------------------------------

    # Epoch averages

    # -----------------------------------------------------



    epoch_losses = {

        key: value / num_batches

        for key, value in running_losses.items()

    }



    epoch_time = (

        time.time() - epoch_start

    )



    print(

        f"Epoch {epoch}/{num_epochs} complete "

        f"| Total loss: "

        f"{epoch_losses['total_loss']:.4f} "

        f"| Time: {epoch_time:.1f}s",

        flush=True,

    )





    # -----------------------------------------------------

    # Save training log

    # -----------------------------------------------------



    with open(

        log_path,

        "a",

        newline="",

    ) as f:



        writer = csv.writer(f)



        writer.writerow([

            epoch,

            epoch_losses["total_loss"],

            epoch_losses["loss_classifier"],

            epoch_losses["loss_box_reg"],

            epoch_losses["loss_mask"],

            epoch_losses["loss_objectness"],

            epoch_losses["loss_rpn_box_reg"],

            epoch_time,

        ])





    # -----------------------------------------------------

    # Save checkpoint

    # -----------------------------------------------------



    checkpoint_path = (

        checkpoint_dir

        / f"epoch_{epoch:02d}.pth"

    )



    torch.save(

        {

            "epoch": epoch,

            "model_state_dict": (

                model.state_dict()

            ),

            "optimizer_state_dict": (

                optimizer.state_dict()

            ),

            "loss": (

                epoch_losses["total_loss"]

            ),

            "seed": SEED,

            "experiment": EXP_NAME,

            "config": exp_config,

        },

        checkpoint_path,

    )



    print(

        "Checkpoint saved:",

        checkpoint_path,

        flush=True,

    )



    print()





print("=" * 70)

print("Training completed.")

print("Training log:", log_path)

print("=" * 70)
