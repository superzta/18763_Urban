# RUIDR: Real-Time Urban Issue Detection and Ranking

A PyTorch implementation of Faster R-CNN for detecting and classifying urban infrastructure issues with support for **10 urban issue classes**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

##  Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Urban Issue Classes](#-urban-issue-classes)
- [Usage](#-usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Inference](#inference)
- [Multi-Class Training](#-multi-class-training)
- [Configuration](#-configuration)
- [Command Reference](#-command-reference)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Authors](#-authors)

---

##  Features

### Core Features
-  **Faster R-CNN** with ResNet-50-FPN backbone
-  **Multi-class training** - Train on any combination of 10 urban issue types
-  **GPU accelerated** - Automatic CUDA detection and usage
-  **Pretrained weights** - ImageNet initialization for faster convergence
-  **Complete pipeline** - Training, testing, and inference modes
-  **Visualization** - Training curves and prediction comparisons

### Training Features
- Automatic checkpointing (best model + periodic saves)
- Learning rate scheduling (StepLR)
- Real-time progress tracking with tqdm
- Comprehensive logging (JSON format)
- Data augmentation support

### Evaluation Features
- mAP@0.5 (mean Average Precision)
- Precision & Recall metrics
- Visual comparison (Ground Truth vs Predictions)
- Per-class performance analysis

### Inference Features
- Single image or batch processing
- Adjustable confidence thresholds
- Bounding box visualization with class labels
- Structured JSON output

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Training (5 minutes)
```bash
python retrain_and_visualize.py
# Press Enter to use default (Broken Road Signs)
# Train for 10 epochs
```

### 3. View Results
- Training curves: `results/training_curves.png`
- Predictions vs Ground Truth: `results/eval_comparison_*.png`
- Metrics: `results/evaluation_results.json`

---

##  Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv rcnn_env
rcnn_env\Scripts\activate

# Linux/Mac
python3 -m venv rcnn_env
source rcnn_env/bin/activate
```

### Step 2: Install PyTorch
**For GPU (Recommended):**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 13.0 (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**For CPU only:**
```bash
pip install torch torchvision
```

### Step 3: Install Other Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python verify_installation.py
```

### Step 5: Download Dataset
The dataset should be organized in the following structure:
```
data/
├── config.yaml                          # Global class definitions (0-9)
├── DamagedRoadSigns/
│   └── DamagedRoadSigns/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       └── test/
├── IllegalParking/
├── Potholes and RoadCracks/
├── ... (other datasets)
```

---

##  Urban Issue Classes

The system supports **10 urban issue classes** defined in `data/config.yaml`:

| ID | Class Name | Dataset Size (Train) | Description |
|----|-----------|---------------------|-------------|
| 0 | Damaged Road issues | 5,667 | Road cracks and surface damage |
| 1 | Pothole Issues | 5,667 | Potholes on roads |
| 2 | Illegal Parking Issues | 57  | Illegally parked vehicles |
| 3 | Broken Road Sign Issues | 2,267 | Damaged or broken signs |
| 4 | Fallen trees | 8,500 | Fallen trees blocking roads |
| 5 | Littering/Garbage | 3,133 | Garbage on public places |
| 6 | Vandalism Issues | 1,703 | Graffiti and vandalism |
| 7 | Dead Animal Pollution | 172 | Dead animals on roads |
| 8 | Damaged concrete structures | 9,315 | Damaged infrastructure |
| 9 | Damaged Electric wires | 7,254 | Damaged electrical poles |

**Default Training:** Class 3 (Broken Road Sign Issues)

---

##  Usage

### Training

#### Option 1: Interactive Training (Recommended for Beginners)
```bash
python retrain_and_visualize.py
```
- Prompts you to select classes interactively
- Default: Class 3 (Broken Road Signs)
- Quick training with visualization

#### Option 2: Command-Line Training
```bash
# Default (single class)
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005

# With custom settings
python rcnn.py --mode train \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.003
```

#### Option 3: Programmatic Training
```python
from rcnn import Config, train

# Single class (default: Broken Road Signs)
config = Config(urban_issue_classes=[3])
config.num_epochs = 50
config.batch_size = 4
model = train(config)

# Multiple classes
config = Config(urban_issue_classes=[0, 1, 3])  # Roads, Potholes, Signs
config.num_epochs = 50
model = train(config)
```

#### Option 4: Training Examples
```bash
python train_examples.py
# Select from predefined scenarios:
# 1. Single class
# 2. Road-related issues
# 3. Infrastructure issues
# 4. Environmental issues
# 5. ALL classes
# 6. Custom selection
```

#### Option 5: Testing Examples
```bash
python test_examples.py
# Select from predefined testing scenarios:
# 1. Single class (Broken Road Signs)
# 2. Road-related issues
# 3. Infrastructure issues
# 4. Environmental issues
# 5. ALL classes
# 6. Custom selection
```

### Testing

Evaluate model performance on test set:

```bash
# Test on single class (default: class 3)
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# Test on specific classes (must match training classes!)
python rcnn.py --mode test --classes 3 --checkpoint checkpoints/best_model.pth
python rcnn.py --mode test --classes 0,1,3 --checkpoint checkpoints/best_model.pth
```

**Important:** The `--classes` should match the classes used during training!

**Output:**
- mAP@0.5, Precision, Recall
- Results saved to `results/evaluation_results.json`
- Visual comparisons: `results/eval_comparison_*.png`

### Inference

#### Single Image
```bash
# Default (class 3)
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --conf-threshold 0.5

# Specify classes
python rcnn.py --mode inference \
    --classes 0,1,3 \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --conf-threshold 0.5
```

#### Batch Processing
```bash
python rcnn.py --mode inference \
    --classes 3 \
    --checkpoint checkpoints/best_model.pth \
    --image-dir path/to/images/ \
    --conf-threshold 0.5
```

#### Programmatic Inference
```python
from rcnn import Config, inference

config = Config(urban_issue_classes=[3])
config.conf_threshold = 0.5

boxes, labels, scores = inference(
    config,
    'checkpoints/best_model.pth',
    'test_image.jpg',
    save_path='output.jpg'
)
```

---

##  Multi-Class Training

### How It Works

The system uses a **3-level class mapping**:

1. **Global Class IDs (0-9)** - Defined in `data/config.yaml`
2. **Local Class IDs** - Used in individual subdataset label files
3. **Model Output Indices** - 0=background, 1=first selected class, etc.

**Example:** Training on classes `[0, 3, 8]`
```
Model outputs:
  0 = background
  1 = Damaged Road issues (global class 0)
  2 = Broken Road Sign Issues (global class 3)
  3 = Damaged concrete structures (global class 8)
```

### Training Scenarios

#### Scenario 1: Single Class (Default)
```python
config = Config(urban_issue_classes=[3])  # Broken Road Signs
config.num_epochs = 30
model = train(config)
```

#### Scenario 2: Road Safety Monitoring
```python
config = Config(urban_issue_classes=[0, 1, 3])
# Roads, Potholes, Signs
config.num_epochs = 40
config.batch_size = 6
model = train(config)
```

#### Scenario 3: Infrastructure Maintenance
```python
config = Config(urban_issue_classes=[3, 8, 9])
# Signs, Concrete, Electrical
config.num_epochs = 35
model = train(config)
```

#### Scenario 4: Environmental Monitoring
```python
config = Config(urban_issue_classes=[4, 5, 7])
# Trees, Garbage, Dead animals
config.num_epochs = 30
model = train(config)
```

#### Scenario 5: Comprehensive City Monitoring
```python
config = Config(urban_issue_classes=list(range(10)))  # All 0-9
config.num_epochs = 60
config.batch_size = 8
model = train(config)
```

### Class Selection Guidelines

| Number of Classes | Recommended Epochs | Batch Size | Learning Rate |
|-------------------|-------------------|------------|---------------|
| 1 class | 20-30 | 4 | 0.005 |
| 2-3 classes | 30-40 | 4-6 | 0.005 |
| 4-6 classes | 40-60 | 6-8 | 0.003-0.005 |
| 7-10 classes | 60-100 | 8-16 | 0.003 |

---

## Configuration

### Hyperparameters

The `Config` class in `rcnn.py` manages all settings:

```python
class Config:
    def __init__(self, urban_issue_classes=[3]):
        # Model
        self.num_classes = len(urban_issue_classes) + 1  # +1 for background
        self.pretrained = True
        
        # Training
        self.batch_size = 4
        self.num_epochs = 50
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.weight_decay = 0.0005
        
        # Learning rate scheduler
        self.lr_scheduler_step_size = 10  # Decay every N epochs
        self.lr_scheduler_gamma = 0.1     # Multiply by this factor
        
        # Data augmentation
        self.horizontal_flip_prob = 0.5
        self.min_size = 640
        self.max_size = 640
        
        # Inference
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = 0.3   # NMS IoU threshold
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Modifying Configuration

**Option 1: Edit directly**
```python
config = Config(urban_issue_classes=[3])
config.num_epochs = 100
config.batch_size = 8
config.learning_rate = 0.003
```

**Option 2: Command-line arguments**
```bash
python rcnn.py --mode train --epochs 100 --batch-size 8 --lr 0.003
```

---

##  Command Reference

### Training Commands

```bash
# Quick test (5 epochs)
python rcnn.py --mode train --epochs 5 --batch-size 2

# Standard training
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005

# High-quality training
python rcnn.py --mode train --epochs 100 --batch-size 8 --lr 0.003

# Low memory (4GB GPU)
python rcnn.py --mode train --epochs 50 --batch-size 1

# Interactive training
python retrain_and_visualize.py

# Example scenarios
python train_examples.py
```

### Testing Commands

```bash
# Test best model (default class 3)
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# Test on specific classes
python rcnn.py --mode test --classes 3 --checkpoint checkpoints/best_model.pth
python rcnn.py --mode test --classes 0,1,3 --checkpoint checkpoints/best_model.pth

# Test specific checkpoint
python rcnn.py --mode test --classes 3 --checkpoint checkpoints/fasterrcnn_epoch_50.pth
```

### Inference Commands

```bash
# Single image (default class 3)
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image image.jpg

# Single image with specific classes
python rcnn.py --mode inference \
    --classes 0,1,3 \
    --checkpoint checkpoints/best_model.pth \
    --image image.jpg

# Batch inference
python rcnn.py --mode inference \
    --classes 3 \
    --checkpoint checkpoints/best_model.pth \
    --image-dir images/

# Lower confidence (more detections)
python rcnn.py --mode inference \
    --classes 3 \
    --checkpoint checkpoints/best_model.pth \
    --image image.jpg \
    --conf-threshold 0.3

# Higher confidence (fewer, confident detections)
python rcnn.py --mode inference \
    --classes 3 \
    --checkpoint checkpoints/best_model.pth \
    --image image.jpg \
    --conf-threshold 0.8
```

### Utility Commands

```bash
# Verify installation
python verify_installation.py

# Complete workflow demo
python complete_workflow_example.py

# Advanced inference examples
python inference_example.py

# Check GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor GPU during training
watch -n 1 nvidia-smi  # Linux/Mac
```

---

##  Project Structure

```
18763_Urban/
├── data/                              # Dataset directory
│   ├── config.yaml                    # Global class definitions (0-9)
│   ├── DamagedRoadSigns/
│   ├── IllegalParking/
│   ├── Potholes and RoadCracks/
│   └── ... (other datasets)
│
├── checkpoints/                       # Model checkpoints (auto-created)
│   ├── best_model.pth                 # Best model
│   └── fasterrcnn_epoch_*.pth         # Periodic checkpoints
│
├── outputs/                           # Inference outputs (auto-created)
│   └── inference_*.jpg
│
├── results/                           # Training/evaluation results (auto-created)
│   ├── training_curves.png            # Loss curves
│   ├── training_history.json          # Detailed logs
│   ├── evaluation_results.json        # mAP, precision, recall
│   └── eval_comparison_*.png          # GT vs Predictions
│
├── rcnn.py                           # Main script (training/testing/inference)
├── retrain_and_visualize.py          # Interactive training script
├── train_examples.py                 # Predefined training scenarios
├── test_examples.py                  # Predefined testing scenarios
├── complete_workflow_example.py      # Complete demo workflow
├── inference_example.py              # Advanced inference examples
├── config_example.py                 # Configuration examples
├── verify_installation.py            # Installation verification
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

##  Examples

### Example 1: Quick Training and Testing
```bash
# Train for 10 epochs
python rcnn.py --mode train --epochs 10 --batch-size 4

# Test the model
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# Or use test examples script
python test_examples.py

# Run inference
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image-dir data/DamagedRoadSigns/DamagedRoadSigns/test/images/
```

### Example 2: Multi-Class Training
```python
from rcnn import Config, train, test

# Train on road-related issues
config = Config(urban_issue_classes=[0, 1, 3])
config.num_epochs = 40
config.batch_size = 6
model = train(config)

# Test
results = test(config, 'checkpoints/best_model.pth')
print(f"mAP: {results['mAP@0.5']:.4f}")
```

### Example 3: Custom Inference Pipeline
```python
from rcnn import Config, get_model
from PIL import Image
from torchvision.transforms import functional as F
import torch

# Load model
config = Config(urban_issue_classes=[3])
model = get_model(config.num_classes, pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(config.device)
model.eval()

# Process image
image = Image.open('test.jpg').convert('RGB')
image_tensor = F.to_tensor(image).unsqueeze(0).to(config.device)

# Get predictions
with torch.no_grad():
    predictions = model(image_tensor)[0]

# Filter by confidence
mask = predictions['scores'] > 0.5
boxes = predictions['boxes'][mask]
labels = predictions['labels'][mask]
scores = predictions['scores'][mask]

print(f"Detected {len(boxes)} objects")
```

---

##  Troubleshooting

### Installation Issues

**Problem: CUDA out of memory**
```bash
# Solution 1: Reduce batch size
python rcnn.py --mode train --batch-size 1

# Solution 2: Use CPU
# Edit rcnn.py Config class:
self.device = torch.device('cpu')
```

**Problem: torch.cuda.is_available() returns False**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem: NotImplementedError: Could not run 'torchvision::nms' with CUDA**
```bash
# Reinstall torchvision with matching CUDA version
pip uninstall torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training Issues

**Problem: Loss not decreasing**
```python
# Lower learning rate
config.learning_rate = 0.001  # or 0.002
```

**Problem: Loss oscillating**
```python
# Lower LR or increase batch size
config.learning_rate = 0.001
config.batch_size = 8
```

**Problem: Training very slow**
```bash
# Check GPU usage
nvidia-smi

# Increase batch size (if GPU allows)
python rcnn.py --mode train --batch-size 8

# Increase num_workers
# Edit Config class:
self.num_workers = 8
```

**Problem: Loss is NaN**
```python
# Much lower learning rate
config.learning_rate = 0.0001
```

### Inference Issues

**Problem: No detections**
```bash
# Lower confidence threshold
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image test.jpg \
    --conf-threshold 0.3
```

**Problem: Too many false positives**
```bash
# Increase confidence threshold
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image test.jpg \
    --conf-threshold 0.7

# Train longer
python rcnn.py --mode train --epochs 100
```

### Dataset Issues

**Problem: Small dataset (e.g., Illegal Parking with only 57 images)**

Solutions:
1. **Train separately with more epochs:**
   ```python
   config = Config(urban_issue_classes=[2])
   config.num_epochs = 100  # More epochs for small dataset
   ```

2. **Use data augmentation:**
   ```python
   config.horizontal_flip_prob = 0.8
   # Add more augmentation in MultiClassUrbanDataset
   ```

3. **Skip class in multi-class training:**
   ```python
   # Exclude class 2 (Illegal Parking)
   config = Config(urban_issue_classes=[0,1,3,4,5,6,7,8,9])
   ```

---


## Important Notes

1. **Class 2 (Illegal Parking)** has very few images (57 train). Consider:
   - Training separately with more epochs
   - Using heavy data augmentation
   - Collecting more data

2. **Multi-class training** requires more epochs and GPU memory

3. **Always check visualizations** (`results/eval_comparison_*.png`) to understand model performance

4. **Class names** are automatically loaded from `data/config.yaml` - no need to manually specify

---

##  Output Files

### Training Outputs

```
checkpoints/
├── best_model.pth              # Best model (lowest loss)
└── fasterrcnn_epoch_N.pth      # Checkpoint at epoch N

results/
├── training_curves.png         # Loss curves visualization
└── training_history.json       # Detailed training logs
```

### Evaluation Outputs

```
results/
├── evaluation_results.json     # mAP, precision, recall
└── eval_comparison_*.png       # GT (green) vs Predictions (red)
```

### Inference Outputs

```
outputs/
└── inference_*.jpg             # Annotated images with bounding boxes
```

---

## Authors

- **Jiadong Zhang** - jiadong2@andrew.cmu.edu
- **Owen Zeng** - ozeng@andrew.cmu.edu
- **Maochuan Lu** - maochual@andrew.cmu.edu

**Course:** 18-794 Pattern Recognition Theory  
**Institution:** Carnegie Mellon University  
**Semester:** Fall 2025

---

##  Acknowledgments

- **Dataset:** Urban Issues Dataset from Kaggle
- **Architecture:** Faster R-CNN with ResNet-50-FPN backbone
- **Framework:** PyTorch and torchvision
- **Course:** 18-794 Pattern Recognition Theory, CMU

---

##  Quick Reference

**Fastest way to get started:**
```bash
pip install -r requirements.txt
python retrain_and_visualize.py
```

**Most common workflow:**
```bash
# Train
python rcnn.py --mode train --epochs 50 --batch-size 4

# Test
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# Inference
python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image test.jpg
```

**Using example scripts:**
```bash
# Training examples
python train_examples.py

# Testing examples
python test_examples.py

# Inference examples
python inference_example.py
```

**Happy detecting! **

---

*Last updated: October 2025*
