# LUS-7Seg: A Multi-Annotation Dataset and Benchmark for Standard Plane Classification and Anatomical Segmentation in Hepatobiliary Ultrasound

Official code repository for the paper **"LUS-7Seg: A Multi-Annotation Dataset and Benchmark for Standard Plane Classification and Anatomical Segmentation in Hepatobiliary Ultrasound"**.

> Hepatobiliary ultrasound (US) diagnosis is highly operator-dependent, and the lack of large-scale, well-annotated datasets hinders the development of automated deep learning solutions. We introduce LUS-7Seg, a multi-annotation dataset comprising 10,507 2D ultrasound images from 2,415 healthy subjects, covering the 7 core hepatobiliary standard planes with expert-verified pixel-level masks for liver parenchyma and gallbladder.

## Key Features
- **Large-scale, multi-vendor dataset** – 10,507 images from 2,415 subjects, acquired on 4 major ultrasound devices.
- **Dual-task annotations** – 7-class standard plane labels + pixel-level liver & gallbladder masks.
- **Comprehensive benchmarks** – Classification (ResNet50, DenseNet121, ViT-B/16) and segmentation (U-Net, nnU-Net).
- **Interpretability analysis** – Multi-threshold alignment IoU to expose shortcut learning in pure classification models.
- **Ready-to-use code** – Training, evaluation, and interpretability scripts organized by model.

## Benchmark Results

We provide comprehensive baselines for both standard plane classification and anatomical segmentation.

**Classification (Test Set)**
| Architecture | Accuracy | Macro F1 | AUC-ROC |
| :--- | :--- | :--- | :--- |
| ResNet50 | 0.9045 | 0.8931 | 0.9807 |
| ViT-B/16 | 0.9040 | 0.8952 | 0.9881 |
| DenseNet121 | 0.9088 | 0.8962 | 0.9828 |

**Segmentation (Original Resolution)**
| Architecture | Class | Mean Dice | Mean HD95 (px) |
| :--- | :--- | :--- | :--- |
| U-Net (ResNet50) | Liver | 0.9262 | 26.9383 |
| nnU-Net v2 | Liver | 0.9360 | 22.9785 |
| U-Net (ResNet50) | Gallbladder | 0.7429 | 12.8061 |
| nnU-Net v2 | Gallbladder | 0.7583 | 10.0411 |

## Dataset
To strictly maintain the integrity of the double-blind peer review process, the full **LUS-7Seg dataset** has been securely reserved on Mendeley Data and will be fully accessible upon official publication. 

🔗 **Reserved Mendeley Data DOI:** `10.17632/cwc37g672d.1` 
*(Note: This DOI link is currently inactive to preserve anonymity and will be activated immediately post-acceptance).*

For detailed information regarding dataset acquisition, preprocessing, and hierarchical directory organization, please refer to **Section 3** of the main manuscript.

**Data structure expected by the code:**

```text
data/
├── Patient_0001/
│   ├── FPH/
│   │   ├── image1.jpg     # cropped 2D ultrasound image
│   │   ├── image1.json    # raw annotation
│   │   └── image1.png     # pixel-level semantic mask 
│   ├── GBH/
│   ├── LHA/
│   ├── LHP/
│   ├── LHV/
│   ├── RH/
│   └── SPH/
├── Patient_0002/
└── ...
```

**Mask color encoding:**
- White (255, 255, 255) → Liver parenchyma
- Green (0, 255, 0)     → Gallbladder
- Black (0, 0, 0)       → Background

Set the environment variable `LUS7SEG_DATA` to point to your `data/` directory.

## Repository Structure

```text
.
├── DenseNet/ # Classification with DenseNet121
├── ResNet50/ # Classification with ResNet50
├── ViT/ # Classification with ViT-B/16
├── UNet/ # Segmentation with ResNet50-U-Net
├── nnUNet/ # Segmentation with nnU-Net v2 (format conversion + evaluation)
├── explain/ # CAM-based interpretability analysis (all three classifiers)
│   ├── cam_utils.py
│   ├── config.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── resnet_analysis.py
│   ├── densenet_analysis.py
│   └── vit_analysis.py
└── requirements.txt 
```

## Installation
```bash
git clone https://github.com/[Your GitHub Username]/LUS-7Seg.git
cd LUS-7Seg
pip install -r requirements.txt
```

For nnU-Net, also install:

```bash
pip install nnunetv2
```

## Environment Variables

The scripts rely on several environment variables. Set them before running:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LUS7SEG_DATA` | Path to the dataset root (containing Patient_0001, ...) | `./data` |
| `MODEL_DIR` | Directory to save trained model checkpoints | `./checkpoints` |
| `EXPLAIN_DIR` | Base directory for interpretability output | `./explain_results` |
| `NNUNET_RAW` | nnU-Net raw data folder | `./nnUNet_raw` |
| `NNUNET_PREDICT_DIR` | nnU-Net prediction output folder | `./nnUNet_results/predictions_ensemble` |
| `RESNET_WEIGHT` | Path to trained ResNet50 weights | `./checkpoints/best_resnet50.pth` |
| `DENSENET_WEIGHT` | Path to trained DenseNet121 weights | `./checkpoints/best_densenet121.pth` |
| `VIT_WEIGHT` | Path to trained ViT weights | `./checkpoints/best_vit.pth` |

You can export them in your shell or modify the defaults in the respective `config.py` files.

## Quick Start

### 1. Classification

Each classifier has its own folder. To train and evaluate, for example, DenseNet121:

```bash
cd DenseNet
python train.py      # trains the model, saves best checkpoint
python evaluate.py   # prints metrics, saves confusion matrix
```

Replace `DenseNet`, `ResNet50` or `ViT` for the other architectures.
**Note:** All classification experiments use the **same patient-level split** (70-10-20) with seed 42.

### 2. Segmentation – U-Net (ResNet50 encoder)

```bash
cd UNet
python train.py
python evaluate.py   # computes metrics at original resolution, saves visualizations
```

### 3. Segmentation – nnU-Net v2

Step-by-step:

```bash
cd nnUNet

# Convert LUS-7Seg data to nnU-Net format (Dataset501_LUS7Seg)
python convert_to_nnunet.py

# Set nnU-Net environment variables
export nnUNet_raw="$PWD/../nnUNet_raw"
export nnUNet_preprocessed="$PWD/../nnUNet_preprocessed"
export nnUNet_results="$PWD/../nnUNet_results"

# Preprocess and train (2D U-Net, 5-fold cross-validation)
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 2d 0  # fold 0 (repeat for folds 1..4)

# Predict on test set
nnUNetv2_predict -d 501 -i $nnUNet_raw/Dataset501_LUS7Seg/imagesTs -o $nnUNet_results/predictions_ensemble -f 0 1 2 3 4 -c 2d

# Evaluate predictions
python evaluate.py
```

*Note:* The `convert_to_nnunet.py` script applies the same patient-level split (seed 42) as the other benchmarks, merging Train+Val as the training set for nnU-Net.

### 4. Interpretability Analysis (CAM & Alignment IoU)

```bash
cd explain
```

Run the analysis for each classifier:

```bash
python resnet_analysis.py
python densenet_analysis.py
python vit_analysis.py
```

Each script generates a multi-threshold Alignment IoU curve (`iou_threshold_curve.png`) inside `explain_results/<model>_results/`.
The analysis uses the **correctly predicted** test samples to compute spatial overlap between CAM heatmaps and the expert masks.


## License

The codebase is released under the [MIT License](LICENSE). 
The LUS-7Seg dataset is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the dataset, provided you give appropriate credit by citing our paper.

## Citation

If you find this dataset or code useful for your research, please cite our paper:
Citation details will be updated upon publication.