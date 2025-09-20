# ChangeFormer: A Transformer-Based Siamese Network for Change Detection

> **Custom Implementation for Cartosat-3 Satellite Data**
> 
> This repository contains a customized implementation of ChangeFormer specifically trained and optimized for **Cartosat-3** satellite imagery change detection tasks.

## 🛰️ Cartosat-3 Implementation Overview

This implementation has been specifically adapted and trained for **Cartosat-3** satellite data with **0.7-0.8m resolution**, providing superior change detection capabilities for high-resolution satellite imagery.

### Key Features of This Implementation:
- **Pre-trained Foundation**: Initialized with LEVIR-CD weights trained on ~10,192 aerial images
- **Multi-Dataset Training**: Fine-tuned on OSCD and Sentinel-2 datasets before final Cartosat-3 optimization  
- **High Resolution Support**: Optimized for 0.7-0.8m resolution Cartosat-3 imagery
- **Enhanced Training Strategies**: Cross-validation with augmentation for improved generalization
- **Dual Backbone Support**: ResNet18 and ResNet50 backbones for stability and performance

### Training Evolution:
1. **LEVIR-CD Pre-training**: Started with robust aerial image foundation ([Pre-trained Weights] " to be done ")
2. **OSCD Fine-tuning**: Initial satellite adaptation (limited success due to resolution constraints & limited data)
3. **Sentinel-2 Experiments**: Tested but resolution limitations affected performance
4. **Cartosat-3 Optimization**: Final implementation with optimal resolution and performance

## 📁 Data Structure and Preparation

### Required Directory Structure:
```
Your_Dataset/
├── A/                  # T1 phase images (numerically ordered: 0000.tif, 0001.tif, ...)
├── B/                  # T2 phase images (numerically ordered: 0000.tif, 0001.tif, ...)
├── label/              # Binary change masks (numerically ordered: 0000.tif, 0001.tif, ...)
└── list/
    ├── train.txt       # Training image names
    └── val.txt         # Validation image names
```

### Data Preparation Script:
Use `data_check.py` for complete data organization - contains 3 separate code sections:
1. **Section 1**: Organize A & B folders with numerical naming
2. **Section 2**: Organize label folder with numerical naming  
3. **Section 3**: Generate train/val split files

### Dataset Configuration:
Add your dataset to `data_config.py`:
```python
elif data_name == 'cartoCustom':  # or your dataset name
    self.label_transform = "norm"
    self.root_dir = './data/cartoCustom/'  # your dataset path
```

## 🏗️ Model Architecture and Components

### Backbone Options:
- **ResNet18**: Lightweight and efficient
- **ResNet50**: More stable performance (recommended for production)

### Key Files:
- `models/`: Contains model architecture, loss functions, and backbone structures
- `datasets/CD_dataset.py`: Custom dataloader implementation
- `utils.py`: Utility functions for data processing

### Loss Functions:
Enhanced loss combinations optimized for change detection:
- **Focal Tversky Loss**: Handles class imbalance effectively
- **Weighted BCE**: Additional support for imbalanced datasets
- **Combined Loss**: Optimized weighting for best performance

## 🚀 Training Options

### Basic Training:
```bash
# Basic training scripts
python train_carto.py      # Standard Cartosat-3 training
python train_custom.py     # Custom dataset training
```

### Cross-Validation Training:
```bash
# Single Fold Cross-Validation
python train_cartoSingleFold.py        # Basic single fold CV
python train_singleFoldUpdated.py      # Enhanced single fold CV

# Multi-Fold Cross-Validation  
python train_newCarto_2fold.py         # 2-fold CV
python train_carto3Fold.py             # 3-fold CV (recommended)
```

### Training Configuration:
Key parameters from argument parser:
- **Backbone**: ResNet18/50 for stability
- **Epochs**: 120 (with early stopping)
- **Loss Function**: Combined Focal Tversky + Weighted BCE
- **Augmentation**: In-memory augmentation with cross-validation
- **Learning Rate**: Cosine warm restarts scheduler
- **Batch Size**: Optimized for memory efficiency

## 🔍 Inference

### Single Pair Inference:
```bash
python cartoInferenceSingle_pair.py   # Process individual image pairs
```

### Batch Inference:
```bash
python cartoInferenceBatch_pair.py    # Process multiple image pairs
```

### Test Data:
- `TEST_DATA/`: Contains unseen data for model evaluation
- Use this folder to validate model performance on new data

## 📊 Results and Visualization

Training results and visualizations are automatically saved in:
- `vis/`: Training progress, validation results, and sample predictions
- Generated every few epochs for monitoring training progress

---

## Original ChangeFormer Documentation

> [A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293)

> [Wele Gedara Chaminda Bandara](https://www.wgcban.com/), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

> Presented at [IGARSS-22](https://www.igarss2022.org/default.php), Kuala Lumpur, Malaysia.

Useful links:
- Paper (published): https://ieeexplore.ieee.org/document/9883686
- Paper (ArXiv): https://arxiv.org/abs/2201.01293
- Presentation (in YouTube): https://www.youtube.com/watch?v=SkiNoTrSmQM

## My other Change Detection repos:

- Change Detection with Denoising Diffusion Probabilistic Models: [DDPM-CD](https://github.com/wgcban/ddpm-cd)
- Semi-supervised Change Detection: [SemiCD](https://github.com/wgcban/SemiCD)
- Unsupervised Change Detection: [Metric-CD](https://github.com/wgcban/Metric-CD)

## Network Architecture
<img width="1003" height="393" alt="image" src="https://github.com/user-attachments/assets/f63ca291-2d91-46e5-83f2-c5ce422d0ea2" />


## Quantitative & Qualitative Results on LEVIR-CD and DSIFN-CD
<img width="1346" height="890" alt="image" src="https://github.com/user-attachments/assets/68de40a1-a938-4b35-bbfb-ca1ed1f3a3fb" />


# Usage
## Requirements

```
Python 3.12.7
pytorch 2.7.0
torchvision 0.23.0
einops  0.8.1
```

- Please see `requirements.txt` for all the other requirements.

## Setting up conda environment: 

Create a virtual ``conda`` environment named ``ChangeFormer`` with the following command:

```bash
conda create --name ChangeFormer --file requirements.txt
conda activate ChangeFormer (or your venv name)
```

## Installation

Clone this repo:

```shell
git clone https://github.com/aman-singh73/changeFormer_CD.git
cd changeFormer_CD
```

## Quick Start on LEVIR dataset

We have some samples from the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) dataset in the folder `samples_LEVIR` for a quick start.

Firstly, you can download our ChangeFormerV6 pretrained model——by [`Github-LEVIR-Pretrained`](https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip). 

Place it in `checkpoints/ChangeFormer_LEVIR/`.

Run a demo to get started as follows:

```python
python demo_LEVIR.py
```

You can find the prediction results in `samples/predict_LEVIR`.

## Quick Start on DSIFN dataset

We have some samples from the [`DSIFN-CD`](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) dataset in the folder `samples_DSIFN` for a quick start.

Download our ChangeFormerV6 pretrained model——by [`Github`](https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip). After downloaded the pretrained model, you can put it in `checkpoints/ChangeFormer_DSIFN/`.

Run the demo to get started as follows:

```python
python demo_DSIFN.py
```

You can find the prediction results in `samples/predict_DSIFN`.

## Training on LEVIR-CD

When we initialy train our ChangeFormer, we initialized some parameters of the network with a model pre-trained on the RGB segmentation (ADE 160k dataset) to get faster convergence.

You can download the pre-trained model [`Github-LEVIR-Pretrained`](https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip).
```
wget https://www.dropbox.com/s/undtrlxiz7bkag5/pretrained_changeformer.pt
```

Then, update the path to the pre-trained model by updating the ``path`` argument in the ``run_ChangeFormer_LEVIR.sh``.
Here:
https://github.com/wgcban/ChangeFormer/blob/a3eca2b1ec5d0d2628ea2e0b6beae85630ba79d4/scripts/run_ChangeFormer_LEVIR.sh#L28

You can find the training script `run_ChangeFormer_LEVIR.sh` in the folder `scripts`. You can run the script file by `sh scripts/run_ChangeFormer_LEVIR.sh` in the command environment.

The detailed script file `run_ChangeFormer_LEVIR.sh` is as follows:

```cmd
#!/usr/bin/env bash

#GPUs
gpus=0

#Set paths
checkpoint_root=/media/lidan/ssd2/ChangeFormer/checkpoints
vis_root=/media/lidan/ssd2/ChangeFormer/vis
data_name=LEVIR


img_size=256    
batch_size=16   
lr=0.0001         
max_epochs=200
embed_dim=256

net_G=ChangeFormerV6        #ChangeFormerV6 is the finalized verion

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
pretrain=/media/lidan/ssd2/ChangeFormer/pretrained_segformer/segformer.b2.512x512.ade.160k.pth

#Train and Validation splits
split=train         #train
split_val=test      #test, val
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

CUDA_VISIBLE_DEVICES=1 python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
```

## Training on DSIFN-CD

Follow the similar procedure mentioned for LEVIR-CD. Use `run_ChangeFormer_DSIFN.sh` in `scripts` folder to train on DSIFN-CD.

## Evaluate on LEVIR

You can find the evaluation script `eval_ChangeFormer_LEVIR.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval_ChangeFormer_LEVIR.sh` in the command environment.

The detailed script file `eval_ChangeFormer_LEVIR.sh` is as follows:

```cmd
#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=ChangeFormerV6 #This is the best version
split=test
vis_root=/media/lidan/ssd2/ChangeFormer/vis
project_name=CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoints_root=/media/lidan/ssd2/ChangeFormer/checkpoints
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Evaluate on DSIFN

Follow the same evaluation procedure mentioned for LEVIR-CD. You can find the evaluation script `eval_ChangeFormer_DSFIN.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval_ChangeFormer_DSIFN.sh` in the command environment.

### Dataset Preparation

## Data structure

```
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

## Links to processed datsets used for train/val/test

You can download the processed `LEVIR-CD` and `DSIFN-CD` datasets by the DropBox through the following here:

- LEVIR-CD-256: [`click here to download`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)
- DSIFN-CD-256: [`click here to download`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)

Since the file sizes are large, I recommed to use command line and cosider downloading the zip file as follows (in linux):

To download LEVIR-CD dataset run following command in linux-terminal:
```cmd
wget https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip
```
To download DSIFN-CD dataset run following command in linux-terminal:
```cmd
wget https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip
```

For your reference, I have also attached the inks to original LEVIR-CD and DSIFN-CD here: [`LEVIR-CD`](https://justchenhao.github.io/LEVIR/) and [`DSIFN-CD`](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset).

### Other useful notes
#### ChangeFormer for multi-class change detection
If you wish to use ChangeFormer for multi-class change detection, you will need to make a few modifications to the existing codebase, which is designed for binary change detection.
1. `run_ChangeFormer_cd.sh`: n_class=8 and make it a hyperparameter to python main.py
2. `models/networks.py`: net = ChangeFormerV6(embed_dim=args.embed_dim, output_nc=args.n_class)
3. `models/basic_model.py`: Comment out: pred_vis = pred * 255, i.e., modifications to visualisation processing
4. `models/trainer.py`: Modify: ConfuseMatrixMeter(n_class=self.n_class)

### License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
