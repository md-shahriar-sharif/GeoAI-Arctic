import json
from pathlib import Path
import numpy as np
class RTS8BandPreprocessor:
    """Clean and normalize the eight bands used by EXP002."""
    def __init__(self, preprocessing_json):
        with open(Path(preprocessing_json), "r") as f:
            config = json.load(f)
        self.band_names = config["bands"]
        self.mean = np.asarray(config["mean"], dtype=np.float32)
        self.std = np.asarray(config["std"], dtype=np.float32)
        if len(self.mean) != 8 or len(self.std) != 8:
            raise ValueError("Expected statistics for exactly 8 bands.")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("Training means contain non-finite values.")
        if not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError(
                "Training standard deviations must be finite and positive."
            )
    @staticmethod
    def _check_shape(image):
        if image.ndim != 3 or image.shape[-1] != 8:
            raise ValueError(f"Expected H x W x 8 image, got {image.shape}")
    def handle_nan(self, image):
        """Replace NaN/Inf values with the corresponding training-band mean."""
        image = np.asarray(image, dtype=np.float32).copy()
        self._check_shape(image)
        # Broadcasting replaces each invalid pixel with its own band's mean.
        invalid = ~np.isfinite(image)
        if invalid.any():
            image[invalid] = np.broadcast_to(self.mean, image.shape)[invalid]
        return image
    def prepare_8band_maskrcnn(self, image):
        """Return a normalized 8 x H x W float32 array for Mask R-CNN."""
        image = self.handle_nan(image)
        image = (image - self.mean) / self.std
        if not np.all(np.isfinite(image)):
            raise ValueError("Non-finite values remain after preprocessing.")
        image = image.transpose(2, 0, 1)
        return np.ascontiguousarray(image, dtype=np.float32)
