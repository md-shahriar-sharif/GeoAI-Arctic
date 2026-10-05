
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class RTSRawDataset(Dataset):
    """
    Raw PyTorch dataset for the 2026 GeoAI Arctic Challenge.

    Responsibilities:
    - Load variable-size H x W x 8 NPZ image chips
    - Load COCO instance annotations
    - Decode individual RTS masks
    - Convert COCO boxes [x, y, w, h] -> [x1, y1, x2, y2]
    - Return image and instance targets

    Deliberately NOT handled here:
    - NaN imputation
    - normalization
    - band selection
    - resizing
    - augmentation
    """

    def __init__(self, release_root, split_csv, fold="train"):
        self.release_root = Path(release_root)
        self.image_dir = self.release_root / "train" / "images"
        self.annotation_file = (
            self.release_root
            / "train"
            / "annotations"
            / "instances_train.json"
        )

        # Read our fixed group-aware split
        split_df = pd.read_csv(split_csv)

        if fold not in {"train", "val"}:
            raise ValueError("fold must be 'train' or 'val'")

        self.df = (
            split_df[split_df["fold"] == fold]
            .copy()
            .reset_index(drop=True)
        )

        # Load COCO annotation database
        self.coco = COCO(str(self.annotation_file))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = int(row["image_id"])
        file_name = Path(row["file_name"]).name
        image_path = self.image_dir / file_name

        # --------------------------------------------------
        # 1. Load RAW 8-band image
        # --------------------------------------------------
        with np.load(image_path) as data:
            image = data["image"].astype(np.float32)

        if image.ndim != 3 or image.shape[2] != 8:
            raise ValueError(
                f"Expected H x W x 8, got {image.shape} "
                f"for {file_name}"
            )

        height, width, _ = image.shape

        # Convert HWC -> CHW for PyTorch.
        # NaNs remain untouched at this stage.
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()

        # --------------------------------------------------
        # 2. Load this image's COCO annotations
        # --------------------------------------------------
        ann_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            iscrowd=None
        )
        annotations = self.coco.loadAnns(ann_ids)

        masks = []
        boxes = []
        labels = []
        areas = []
        crowds = []

        for ann in annotations:

            # Decode compressed COCO RLE -> binary mask
            mask = self.coco.annToMask(ann).astype(np.uint8)

            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask/image mismatch for image_id={image_id}: "
                    f"mask={mask.shape}, image={(height, width)}"
                )

            # COCO bbox format:
            # [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Torchvision detection format:
            # [x1, y1, x2, y2]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(mask)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann["area"]))
            crowds.append(int(ann.get("iscrowd", 0)))

        # --------------------------------------------------
        # 3. Convert targets to tensors
        # --------------------------------------------------
        if len(masks) > 0:

            masks = torch.as_tensor(
                np.stack(masks),
                dtype=torch.uint8
            )

            boxes = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.as_tensor(
                labels,
                dtype=torch.int64
            )

            areas = torch.as_tensor(
                areas,
                dtype=torch.float32
            )

            crowds = torch.as_tensor(
                crowds,
                dtype=torch.int64
            )

        else:
            masks = torch.zeros(
                (0, height, width),
                dtype=torch.uint8
            )

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

            areas = torch.zeros(
                (0,),
                dtype=torch.float32
            )

            crowds = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(
                image_id,
                dtype=torch.int64
            ),
            "area": areas,
            "iscrowd": crowds,
        }

        return image, target
