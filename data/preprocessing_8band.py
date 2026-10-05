import json

from pathlib import Path



import numpy as np





class RTS8BandPreprocessor:

    """

    Preprocessing for EXP002: 8-band Mask R-CNN.



    Band order:

        0 - red

        1 - green

        2 - blue

        3 - ndvi

        4 - relative_elevation

        5 - shaded_relief

        6 - nir

        7 - ndwi



    Policy:

        1. Replace non-finite values using training-fold band means.

        2. Z-score each band using training-fold mean/std.

        3. Convert HWC -> CHW.



    All statistics come from the fixed 605-image training fold.

    """



    def __init__(self, preprocessing_json):



        preprocessing_json = Path(preprocessing_json)



        with open(preprocessing_json, "r") as f:

            config = json.load(f)



        self.band_names = config["bands"]



        self.mean = np.asarray(

            config["mean"],

            dtype=np.float32,

        )



        self.std = np.asarray(

            config["std"],

            dtype=np.float32,

        )



        if len(self.mean) != 8 or len(self.std) != 8:

            raise ValueError(

                "Expected statistics for exactly 8 bands."

            )



        if np.any(~np.isfinite(self.mean)):

            raise ValueError(

                "Training means contain non-finite values."

            )



        if (

            np.any(~np.isfinite(self.std))

            or np.any(self.std <= 0)

        ):

            raise ValueError(

                "Training standard deviations must be "

                "finite and positive."

            )





    def handle_nan(self, image):

        """

        Replace NaN/Inf values with the corresponding

        training-fold band mean.



        Input:

            H x W x 8 float array



        Returns:

            H x W x 8 float32 array

        """



        image = np.asarray(

            image,

            dtype=np.float32,

        ).copy()



        if image.ndim != 3 or image.shape[2] != 8:

            raise ValueError(

                f"Expected H x W x 8 image, got {image.shape}"

            )



        for band_idx in range(8):



            invalid = ~np.isfinite(

                image[:, :, band_idx]

            )



            if np.any(invalid):



                image[:, :, band_idx][invalid] = (

                    self.mean[band_idx]

                )



        return image





    def prepare_8band_maskrcnn(self, image):

        """

        Prepare all eight challenge bands for EXP002.



        Input:

            H x W x 8 NumPy array



        Returns:

            8 x H x W float32 NumPy array

        """



        # 1. NaN / Inf handling

        image = self.handle_nan(image)



        # 2. Per-band z-score using training-only statistics

        image = (

            image - self.mean[None, None, :]

        ) / self.std[None, None, :]



        # Safety check

        if not np.all(np.isfinite(image)):

            raise ValueError(

                "Non-finite values remain after "

                "8-band preprocessing."

            )



        # 3. HWC -> CHW

        image = np.transpose(

            image,

            (2, 0, 1),

        )



        return np.ascontiguousarray(

            image,

            dtype=np.float32,

        )
