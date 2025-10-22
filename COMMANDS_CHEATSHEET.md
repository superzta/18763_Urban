# Commands Cheat Sheet

Quick reference for all RUIDR Faster R-CNN commands.

## Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py

# Run complete demo
python complete_workflow_example.py
```

## Training Commands

```bash
# Quick test (5-10 minutes)
python rcnn.py --mode train --epochs 5 --batch-size 2

# Standard training (30-60 minutes)
python rcnn.py --mode train --epochs 50 --batch-size 4

# High-quality training (2-4 hours)
python rcnn.py --mode train --epochs 100 --batch-size 8 --lr 0.003

# Low memory training (for 4GB GPU)
python rcnn.py --mode train --epochs 50 --batch-size 1

# Custom learning rate
python rcnn.py --mode train --epochs 50 --lr 0.001

# Programmatic training
python train_example.py
```

## Testing Commands

```bash
# Test on test set
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# Test with different checkpoint
python rcnn.py --mode test --checkpoint checkpoints/fasterrcnn_epoch_50.pth
```

## Inference Commands

```bash
# Single image
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg

# Batch inference
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image-dir path/to/images/

# Lower confidence threshold (more detections)
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image image.jpg \
  --conf-threshold 0.3

# Higher confidence threshold (fewer, confident detections)
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image image.jpg \
  --conf-threshold 0.8

# Advanced inference examples
python inference_example.py
```

## Dataset Commands

```bash
# Use different dataset
python rcnn.py --mode train \
  --data-root "data/Damaged concrete structures/Damaged concrete structures" \
  --data-yaml "data/Damaged concrete structures/data.yaml"

# Use custom paths
python rcnn.py --mode train \
  --data-root /path/to/dataset \
  --data-yaml /path/to/data.yaml
```

## Hyperparameter Tuning

```bash
# Experiment with learning rates
python rcnn.py --mode train --epochs 30 --lr 0.001  # Low
python rcnn.py --mode train --epochs 30 --lr 0.005  # Medium
python rcnn.py --mode train --epochs 30 --lr 0.01   # High

# Experiment with batch sizes
python rcnn.py --mode train --epochs 30 --batch-size 2
python rcnn.py --mode train --epochs 30 --batch-size 4
python rcnn.py --mode train --epochs 30 --batch-size 8

# Combine multiple parameters
python rcnn.py --mode train \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.003 \
  --conf-threshold 0.6
```

## Output Management

```bash
# List checkpoints
ls -lh checkpoints/

# View results
ls results/
cat results/training_history.json
cat results/evaluation_results.json

# View outputs
ls outputs/

# Clean outputs (be careful!)
rm -rf checkpoints/*
rm -rf outputs/*
rm -rf results/*
```

## GPU Commands

```bash
# Check GPU availability
nvidia-smi

# Monitor GPU during training
watch -n 1 nvidia-smi

# Check CUDA in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Troubleshooting Commands

```bash
# Verify installation
python verify_installation.py

# Test with minimal resources
python rcnn.py --mode train --epochs 2 --batch-size 1

# Check Python packages
pip list | grep torch
pip list | grep torchvision

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Workflows

### Complete First-Time Workflow
```bash
# 1. Setup
pip install -r requirements.txt
python verify_installation.py

# 2. Quick test
python rcnn.py --mode train --epochs 5 --batch-size 2

# 3. Check results
ls checkpoints/
cat results/training_history.json

# 4. Test model
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# 5. Run inference
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image data/DamagedRoadSigns/DamagedRoadSigns/test/images/*.jpg
```

### Production Training Workflow
```bash
# 1. Full training
python rcnn.py --mode train --epochs 100 --batch-size 8 --lr 0.005

# 2. Evaluate
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# 3. Batch inference
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image-dir test_images/

# 4. Review results
cat results/evaluation_results.json
```

### Experimentation Workflow
```bash
# Try different configurations
for lr in 0.001 0.005 0.01; do
  python rcnn.py --mode train --epochs 30 --lr $lr
  python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
done

# Compare results
ls -lt results/
```

## All Command-Line Arguments

### Training Mode
```bash
--mode train                  # Training mode
--epochs N                    # Number of epochs
--batch-size N               # Batch size
--lr FLOAT                   # Learning rate
--data-root PATH             # Dataset root directory
--data-yaml PATH             # Path to data.yaml
--checkpoint-dir PATH        # Checkpoint save directory
--output-dir PATH            # Output directory
```

### Test Mode
```bash
--mode test                   # Test mode
--checkpoint PATH            # Model checkpoint path
--data-root PATH             # Dataset root directory
--data-yaml PATH             # Path to data.yaml
--conf-threshold FLOAT       # Confidence threshold
```

### Inference Mode
```bash
--mode inference             # Inference mode
--checkpoint PATH            # Model checkpoint path
--image PATH                 # Single image path
--image-dir PATH             # Image directory path
--conf-threshold FLOAT       # Confidence threshold
--output-dir PATH            # Output directory
```

## Example Scripts

```bash
# Training example
python train_example.py

# Inference examples
python inference_example.py

# Complete workflow
python complete_workflow_example.py

# Installation check
python verify_installation.py
```

## File Locations

```bash
# Code files
rcnn.py                      # Main script
train_example.py             # Training example
inference_example.py         # Inference example
complete_workflow_example.py # Full demo

# Documentation
README.md                    # Full documentation
QUICKSTART.md               # Quick start guide
GETTING_STARTED.md          # Getting started
PROJECT_SUMMARY.md          # Project overview
COMMANDS_CHEATSHEET.md      # This file
config_example.py           # Config examples

# Data
data/                        # Datasets
data/DamagedRoadSigns/      # Road signs dataset

# Outputs
checkpoints/                 # Model checkpoints
outputs/                     # Inference outputs
results/                     # Training/test results
```

## Common Parameter Values

### Learning Rate
- `0.001` - Low (safe, slower convergence)
- `0.005` - Standard (recommended)
- `0.01` - High (faster, may be unstable)

### Batch Size
- `1-2` - For 4GB GPU or CPU
- `4` - Standard for 8GB GPU
- `8-16` - For 12GB+ GPU

### Epochs
- `5-10` - Quick testing
- `50` - Standard training
- `100+` - High-quality training

### Confidence Threshold
- `0.3` - Low (more detections, more false positives)
- `0.5` - Standard (balanced)
- `0.7-0.9` - High (fewer, confident detections)

## Pro Tips

```bash
# Save training output to log file
python rcnn.py --mode train --epochs 50 | tee training.log

# Run in background (Linux/Mac)
nohup python rcnn.py --mode train --epochs 100 &

# Time your training
time python rcnn.py --mode train --epochs 10

# Check model size
ls -lh checkpoints/best_model.pth

# Count images in dataset
find data/DamagedRoadSigns/DamagedRoadSigns/train/images -name "*.jpg" | wc -l
```

## Windows-Specific Commands

```powershell
# Activate virtual environment
venv\Scripts\activate

# List checkpoints
dir checkpoints

# View file content
type results\training_history.json

# Check GPU
nvidia-smi
```

## Linux/Mac-Specific Commands

```bash
# Activate virtual environment
source venv/bin/activate

# List checkpoints
ls -lh checkpoints/

# View file content
cat results/training_history.json

# Monitor training
tail -f training.log

# Run in background
nohup python rcnn.py --mode train --epochs 100 > train.log 2>&1 &
```

---

**Quick Start:**
```bash
pip install -r requirements.txt
python verify_installation.py
python rcnn.py --mode train --epochs 5 --batch-size 2
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image test.jpg
```

**Happy coding! 🚀**

