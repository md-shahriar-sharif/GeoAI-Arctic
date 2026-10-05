from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
class RTSRawDataset(Dataset):
    """Load raw 8-band RTS chips and COCO instance targets."""
    def __init__(self, release_root, split_csv, fold="train"):
        if fold not in {"train", "val"}:
            raise ValueError("fold must be 'train' or 'val'")
        root = Path(release_root)
        self.image_dir = root / "train" / "images"
        self.coco = COCO(
            str(root / "train" / "annotations" / "instances_train.json")
        )
        split = pd.read_csv(split_csv)
        self.samples = split[split["fold"] == fold].reset_index(drop=True)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image_id = int(row["image_id"])
        image = self._load_image(Path(row["file_name"]).name)
        _, height, width = image.shape
        target = self._load_target(image_id, height, width)
        return image, target
    def _load_image(self, file_name):
        path = self.image_dir / file_name
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
        if image.ndim != 3 or image.shape[-1] != 8:
            raise ValueError(
                f"Expected H x W x 8, got {image.shape} for {file_name}"
            )
        # Raw values stay untouched here; preprocessing happens separately.
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float()
    def _load_target(self, image_id, height, width):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        boxes, labels, masks, areas, crowds = [], [], [], [], []
        for ann in annotations:
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask/image mismatch for image_id={image_id}: "
                    f"mask={mask.shape}, image={(height, width)}"
                )
            x, y, w, h = map(float, ann["bbox"])
            box = [x, y, x + w, y + h]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            labels.append(int(ann["category_id"]))
            masks.append(mask)
            areas.append(float(ann["area"]))
            crowds.append(int(ann.get("iscrowd", 0)))
        if masks:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)
            crowds = torch.tensor(crowds, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty(0, dtype=torch.int64)
            masks = torch.empty((0, height, width), dtype=torch.uint8)
            areas = torch.empty(0, dtype=torch.float32)
            crowds = torch.empty(0, dtype=torch.int64)
        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": areas,
            "iscrowd": crowds,
        }
