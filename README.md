# GeoAI Arctic RTS — 8-Band Mask R-CNN Baseline



This repository contains the EXP002 baseline developed for retrogressive thaw slump (RTS) instance segmentation using the 2026 GeoAI Arctic Challenge dataset.



The repository is intentionally limited to the baseline model, preprocessing, training, evaluation, and diagnostic code so that the baseline can be reviewed independently from later experimental modifications.



## Model



The baseline uses Mask R-CNN with a ResNet-50 Feature Pyramid Network (FPN).



The standard 3-channel input convolution is adapted to accept eight remote-sensing channels:



1. Red

2. Green

3. Blue

4. NDVI

5. Relative Elevation

6. Shaded Relief

7. NIR

8. NDWI



Pretrained RGB convolution weights are retained. The five additional channels are initialized using the mean of the pretrained RGB filters.



Model flow:



8-band image -> ResNet-50 -> FPN -> RPN -> RoIAlign -> box classification/regression + mask prediction -> RTS instances



## Repository Structure



- `data/dataset.py` — dataset loader

- `data/preprocessing_8band.py` — 8-band preprocessing

- `model/model_8band.py` — 8-band Mask R-CNN model

- `train/train_exp002_8band.py` — baseline training

- `evaluation/evaluate_exp002_all_epochs.py` — checkpoint evaluation

- `evaluation/diagnose_exp002_small_rts.py` — small-RTS diagnostics

- `evaluation/diagnose_exp002_roi_stage.py` — RoI-stage diagnostics

- `data/preprocessing_v1.json` — training-fold preprocessing statistics

- `config/exp002_reference.json` — EXP002 reference configuration

- `data/split_v1_groupaware.csv` — fixed group-aware train/validation split

- `results/EXP002_results.md` — baseline results and notes



## Challenge Data



The GeoAI Arctic Challenge imagery and annotations are not included in this repository.



The code expects access to the challenge `competition_release` directory.



Set its location with:



    export GEOAI_ARCTIC_DATA=/path/to/competition_release



If the environment variable is not set, the scripts look for `competition_release/` inside the repository root.



## Validation Split



The fixed group-aware split contains:



| Split | Images | RTS Instances |

|---|---:|---:|

| Training | 605 | 1,426 |

| Validation | 151 | 357 |

| Total | 756 | 1,783 |



The split was designed to prevent detected high-similarity image groups from crossing between training and validation.



## Preprocessing



The baseline preprocessing performs:



1. NaN/Inf replacement using training-fold band means.

2. Per-band z-score normalization using training-fold statistics.

3. Conversion of the 8-band image to PyTorch CHW format.



The exact preprocessing statistics are stored in `data/preprocessing_v1.json`.



## Training Configuration



The baseline was trained using:



- Batch size: 2

- Epochs: 10

- Optimizer: SGD

- Learning rate: 0.005

- Momentum: 0.9

- Weight decay: 0.0005

- Random seed: 42



Tested environment:



- Python 3.11.16

- PyTorch 2.11.0

- torchvision 0.26.0

- CUDA 12.8

- NVIDIA A100-SXM4-80GB



## Training



From the repository root:



    python train/train_exp002_8band.py



## Evaluation



Evaluate the EXP002 checkpoints with:



    python evaluation/evaluate_exp002_all_epochs.py



Baseline validation results:



| Metric | Score |

|---|---:|

| Mask AP | 0.487 |

| AP50 | 0.791 |

| AP75 | 0.578 |

| AP-small | 0.351 |

| AP-medium | 0.511 |

| AP-large | 0.593 |



## Small-RTS Diagnostics



Run:



    python evaluation/diagnose_exp002_small_rts.py



and:



    python evaluation/diagnose_exp002_roi_stage.py



For the extreme-small 10–49 px² GT group, the baseline showed:



- RPN recall@0.50: 0.688

- Final recall@0.50: 0.125



This result motivated later research into where very small RTS instances are lost between proposal generation and final prediction.



## Scope



This repository contains EXP002 only.



Later experiments involving context RoI extraction, tiny-object magnification, high-resolution FPN enhancement, attention, custom loss weighting, and combinations of these methods are intentionally excluded from this baseline repository.



Model checkpoints, raw challenge imagery, experiment outputs, and competition submissions are not tracked in Git.

