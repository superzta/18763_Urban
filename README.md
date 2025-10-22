# RUIDR: Real-Time Urban Issue Detection and Ranking

A PyTorch implementation of Faster R-CNN for detecting and classifying urban issues such as damaged road signs, concrete structures, electrical poles, and more.

## Project Overview

This project aims to automate the detection and assessment of urban infrastructure issues using computer vision and deep learning. The system can:
- Detect multiple types of urban issues from images/video
- Classify detected objects (Damaged vs Healthy)
- Provide bounding box localizations
- Run inference in real-time on dashcam footage

## Dataset

We use the **Urban Issues Dataset** in YOLO format with 10 issue categories:
- Damaged Road issues (Road cracks)
- Pothole Issues
- Illegal Parking Issues
- Broken Road Sign Issues
- Fallen trees
- Littering/Garbage on Public Places
- Vandalism Issues (Graffiti)
- Dead Animal Pollution
- Damaged concrete structures
- Damaged Electric wires and poles

### Download the Dataset

```bash
curl -L -o ~/Downloads/urban-issues-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/akinduhiman/urban-issues-dataset
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd 18763_Urban
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, install PyTorch with CUDA:
```bash
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Training

Train the Faster R-CNN model from scratch:

```bash
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 0.005)
- `--data-root`: Root directory of dataset (default: data/DamagedRoadSigns/DamagedRoadSigns)
- `--data-yaml`: Path to data.yaml file
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints/)

**Output:**
- Model checkpoints saved in `checkpoints/` every 5 epochs
- Best model saved as `checkpoints/best_model.pth`
- Training curves saved in `results/training_curves.png`
- Training history saved in `results/training_history.json`

### Testing

Evaluate the trained model on test set:

```bash
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
```

**Output:**
- mAP@0.5, Precision, Recall metrics
- Evaluation results saved in `results/evaluation_results.json`

### Inference

#### Single Image Inference

```bash
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/test_image.jpg \
  --conf-threshold 0.5
```

#### Batch Inference on Directory

```bash
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image-dir path/to/images/ \
  --conf-threshold 0.5
```

**Inference Parameters:**
- `--conf-threshold`: Confidence threshold for detections (default: 0.5)
- `--image`: Path to single input image
- `--image-dir`: Directory containing multiple images

**Output:**
- Annotated images with bounding boxes saved in `outputs/`
- Detection results printed to console

## Configuration

You can easily modify hyperparameters by editing the `Config` class in `rcnn.py` or passing command-line arguments:

### Key Hyperparameters

```python
# Model
num_classes = 3  # background + 2 classes (Damage, Healthy)
pretrained = True  # Use pretrained ResNet50 backbone

# Training
batch_size = 4
num_epochs = 50
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0005

# Learning rate scheduler
lr_scheduler_step_size = 10  # Decay LR every N epochs
lr_scheduler_gamma = 0.1     # Multiply LR by this factor

# Inference
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.3   # NMS IoU threshold
```

## Project Structure

```
18763_Urban/
├── data/                           # Dataset directory
│   ├── DamagedRoadSigns/
│   ├── Damaged concrete structures/
│   ├── DamagedElectricalPoles/
│   └── ...
├── checkpoints/                    # Saved model checkpoints
├── outputs/                        # Inference output images
├── results/                        # Training/evaluation results
├── rcnn.py                        # Main training/testing script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Model Architecture

We use **Faster R-CNN** with a **ResNet-50-FPN** backbone:
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **RPN**: Region Proposal Network for generating object proposals
- **ROI Head**: Fast R-CNN head for classification and bounding box regression
- **Pretrained**: ImageNet pretrained weights (can be disabled)

## Features

**Easy to use**: Simple command-line interface  
**Flexible**: Easily adjustable hyperparameters  
**Comprehensive**: Train, test, and inference modes  
**Production-ready**: Best practices in code structure  
**GPU accelerated**: Automatic GPU detection and usage  
**Visualization**: Training curves and detection visualizations  
**Checkpointing**: Automatic model saving and resuming  
**Evaluation**: mAP, Precision, Recall metrics  

## Performance Monitoring

Training curves are automatically generated and saved to `results/training_curves.png`:
- Total Loss vs Epoch
- Classifier Loss vs Epoch
- Box Regression Loss vs Epoch

## Tips for Better Performance

1. **Increase training epochs**: 50-100 epochs usually work well
2. **Adjust learning rate**: Start with 0.005, decrease if loss oscillates
3. **Batch size**: Larger batch (8-16) if GPU memory allows
4. **Data augmentation**: Enable horizontal flips in Config
5. **Fine-tuning**: Start with pretrained=True for faster convergence

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (try 2 or 1)
- Reduce `num_workers` to 0 or 2
- Use smaller image sizes

### Slow Training
- Increase `num_workers` (4-8)
- Enable GPU if available
- Use mixed precision training (add to future)

### Low mAP
- Train for more epochs
- Adjust confidence threshold
- Check data quality and annotations
- Try different learning rates


## Authors

- Jiadong Zhang (jiadong2@andrew.cmu.edu)
- Owen Zeng (ozeng@andrew.cmu.edu)
- Maochuan Lu (maochual@andrew.cmu.edu)

## Acknowledgments

- Course: 18-794 Pattern Recognition Theory
- Institution: Carnegie Mellon University
- Dataset: Urban Issues Dataset from Kaggle