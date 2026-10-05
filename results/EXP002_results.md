
# EXP002 — 8-Band Mask R-CNN Baseline



## Model



Mask R-CNN with a ResNet-50-FPN backbone adapted from 3-channel RGB input to 8-channel remote-sensing input.



Input channels:



1. Red

2. Green

3. Blue

4. NDVI

5. Relative Elevation

6. Shaded Relief

7. NIR

8. NDWI



The pretrained RGB filters are retained. The five additional input channels are initialized from the mean of the pretrained RGB filters.



## Data Split



Group-aware validation split:



| Split | Images | RTS Instances |

|---|---:|---:|

| Training | 605 | 1,426 |

| Validation | 151 | 357 |

| Total | 756 | 1,783 |



The split was designed to prevent detected high-similarity image groups from crossing between training and validation.



## Training Configuration



- Batch size: 2

- Epochs: 10

- Optimizer: SGD

- Learning rate: 0.005

- Momentum: 0.9

- Weight decay: 0.0005

- Random seed: 42



## Validation Results



| Metric | Result |

|---|---:|

| Mask AP | 0.487 |

| AP50 | 0.791 |

| AP75 | 0.578 |

| AP-small | 0.351 |

| AP-medium | 0.511 |

| AP-large | 0.593 |



## Tiny-RTS Diagnostic



For the extreme-small 10–49 px² GT group:



- RPN recall at IoU 0.50: 0.688

- Final recall at IoU 0.50: 0.125



This indicates that a substantial fraction of extreme-small RTS receive useful RPN proposals but are subsequently lost during downstream processing.



## Purpose



EXP002 is the frozen baseline used as the control for later architecture and training experiments.

