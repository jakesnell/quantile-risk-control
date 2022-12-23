# Quantile Risk Control

Code for paper "Quantile Risk Control: A Flexible Framework for Bounding the Probability of High-Loss Predictions"

Our method provides distribution-free guarantees for a range of loss metrics and weighting functions.

To initialize the conda environment, run the following commands:

    make env-init
    conda activate var_control

Requires python >= 3.10.  Also, you must install the 
<a href="https://github.com/mosco/crossing-probability/blob/master/setup.py">crossing-probability</a>
library.

### Source Data and Models

#### MS-COCO

- MS-COCO is available for download at https://cocodataset.org/
- The TResNet model can be obtained at https://github.com/Alibaba-MIIL/TResNet

#### Nursery

- Dataset can be downloaded from https://archive.ics.uci.edu/ml/datasets/nursery

#### CLIP/CIFAR-100

- CLIP model and CIFAR-100 dataset are available on Huggingface: 
  - https://huggingface.co/datasets/cifar100
  - https://huggingface.co/openai/clip-vit-base-patch32

#### Go Emotions

- Model and dataset are available on Huggingface:
  - https://huggingface.co/datasets/go_emotions
  - https://huggingface.co/bhadresh-savani/bert-base-go-emotion

### Running Experiments

Run the below command to reproduce all of our experiments

    bash paper_experiments.sh
    
For each section, the commands are:

#### Section 5.1
    
    python scripts/multi_experiments.py balanced_accuracy --dataset=coco
    python scripts/multi_experiments.py balanced_accuracy --dataset=go_emotions
    python scripts/multi_experiments.py balanced_accuracy --dataset=clip_cifar_100


#### Section 5.2

    python scripts/interval_experiments.py balanced_accuracy --dataset=clip_cifar_100 --beta_lo=0.6 --beta_hi=0.9 --grid_size=50  --fixed_pred
    python scripts/interval_experiments.py balanced_accuracy --dataset=go_emotions --beta_lo=0.6 --beta_hi=0.9 --grid_size=100 --fixed_pred
    python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.85 --beta_hi=0.95 --grid_size=10 --fixed_pred
    python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.9 --beta_hi=1.0 --grid_size=50 --fixed_pred
    # Fairness
    python scripts/fair_interval_experiments.py custom_loss --dataset=nursery --beta_lo=0.9 --beta_hi=1.0 --grid_size=50 --fixed_pred


#### Section 5.3

    python scripts/mean_experiments.py balanced_accuracy --dataset=coco
    python scripts/var_experiments.py balanced_accuracy --dataset=coco
    python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.85 --beta_hi=0.95 --grid_size=10
    python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.9 --beta_hi=1.0 --grid_size=50

