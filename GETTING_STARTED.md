# Getting Started with RUIDR Faster R-CNN

## Quick Setup (5 Minutes)

### 1. Verify Your Environment

```bash
python verify_installation.py
```

This will check:
- Python version
- Required packages
- CUDA availability
- Dataset presence

### 2. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

For GPU support (highly recommended):
```bash
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Dataset Structure

Your dataset should look like this:
```
data/DamagedRoadSigns/DamagedRoadSigns/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Your First Training Run (10 Minutes)

### Option 1: Quick Test (Recommended First)

```bash
python rcnn.py --mode train --epochs 5 --batch-size 2
```

This will:
- Train for 5 epochs (~5 minutes)
- Use batch size of 2 (works on most GPUs)
- Save checkpoints to `checkpoints/`
- Generate training curves in `results/`

### Option 2: Full Training

```bash
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
```

This will:
- Train for 50 epochs (~30-60 minutes)
- Use standard hyperparameters
- Save best model as `checkpoints/best_model.pth`

### Option 3: Use the Example Script

```bash
python train_example.py
```

This runs training with well-documented configuration.

## Testing Your Model

After training, evaluate performance:

```bash
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
```

You'll see:
- mAP@0.5 (mean Average Precision)
- Precision
- Recall
- Total detections

Results saved to: `results/evaluation_results.json`

## Running Inference

### Single Image

```bash
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image data/DamagedRoadSigns/DamagedRoadSigns/test/images/sample.jpg
```

Output: Annotated image in `outputs/` with bounding boxes

### Multiple Images (Batch Processing)

```bash
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image-dir data/DamagedRoadSigns/DamagedRoadSigns/test/images/
```

### Adjust Detection Threshold

```bash
# More detections (lower confidence threshold)
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image test.jpg \
  --conf-threshold 0.3

# Fewer, more confident detections
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image test.jpg \
  --conf-threshold 0.7
```

## Understanding Results

### Training Outputs

**Checkpoints (`checkpoints/` folder):**
- `best_model.pth` - Best model (lowest loss)
- `fasterrcnn_epoch_N.pth` - Checkpoint at epoch N

**Results (`results/` folder):**
- `training_curves.png` - Visualize loss over time
- `training_history.json` - Detailed training logs

### What to Look For in Training Curves

**Good Training:**
- Steady decrease in all losses
- No sudden spikes
- Smooth curves

**Problems:**
- **Flat line:** Learning rate too low
- **Oscillating:** Learning rate too high or batch size too small
- **Not decreasing:** Model may need more epochs or data issues

### Evaluation Metrics

**mAP (mean Average Precision):**
- > 0.80: Excellent
- 0.60 - 0.80: Good
- 0.40 - 0.60: Fair (needs improvement)
- < 0.40: Poor (check data quality, train longer)

**Precision:** Of all detections, how many are correct?
**Recall:** Of all ground truth objects, how many were detected?

## Common Workflows

### Workflow 1: Quick Experiment
```bash
# 1. Quick training
python rcnn.py --mode train --epochs 10 --batch-size 2

# 2. Test
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# 3. Inference
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image test.jpg
```

### Workflow 2: Full Training Pipeline
```bash
# 1. Full training
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005

# 2. Evaluate
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth

# 3. Batch inference
python rcnn.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --image-dir test_images/
```

### Workflow 3: Hyperparameter Tuning
```bash
# Try different learning rates
python rcnn.py --mode train --epochs 30 --lr 0.001
python rcnn.py --mode train --epochs 30 --lr 0.005
python rcnn.py --mode train --epochs 30 --lr 0.01

# Compare results in results/training_curves.png
```

## Customization

### Change Dataset

Edit `rcnn.py` Config class or use command-line args:

```bash
python rcnn.py --mode train \
  --data-root "data/Damaged concrete structures/Damaged concrete structures" \
  --data-yaml "data/Damaged concrete structures/data.yaml" \
  --epochs 50
```

### Adjust Hyperparameters

All major hyperparameters can be set via command-line:

```bash
python rcnn.py --mode train \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.003 \
  --conf-threshold 0.5
```

Or edit the `Config` class in `rcnn.py`:

```python
class Config:
    def __init__(self):
        self.num_epochs = 100      # Your value
        self.batch_size = 8        # Your value
        self.learning_rate = 0.003 # Your value
        # ... etc
```

## Troubleshooting

### Problem: CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
python rcnn.py --mode train --batch-size 1
```

**Solution 2:** Use CPU (slower but works)
```python
# In rcnn.py Config class:
self.device = torch.device('cpu')
```

### Problem: Training is Very Slow

**Checks:**
1. Is GPU being used? Check with `nvidia-smi`
2. Is CUDA-enabled PyTorch installed?
3. Is `num_workers` too low? Try 4-8

**Solution:**
```bash
# Ensure you're using GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Low mAP

**Possible causes:**
1. Not enough training epochs
2. Learning rate too high/low
3. Data quality issues
4. Incorrect annotations

**Solutions:**
1. Train longer (100+ epochs)
2. Try different learning rates (0.001, 0.005, 0.01)
3. Check dataset manually
4. Review a few training images

### Problem: No Detections During Inference

**Solutions:**
1. Lower confidence threshold:
   ```bash
   --conf-threshold 0.3
   ```
2. Check if model trained properly (view training curves)
3. Verify image format and quality

## Next Steps

After getting started:

1. **Review training curves** - `results/training_curves.png`
2. **Check evaluation metrics** - Run test mode
3. **Visualize predictions** - Run inference on test images
4. **Tune hyperparameters** - Experiment with different settings
5. **Train longer** - 50-100 epochs for production
6. **Try different datasets** - Concrete structures, electrical poles, etc.

## Advanced Usage

### Programmatic Training

See `train_example.py` for a complete example:

```python
from rcnn import Config, train

config = Config()
config.num_epochs = 50
config.batch_size = 4
config.learning_rate = 0.005

model = train(config)
```

### Batch Inference with Results

See `inference_example.py` for advanced inference:

```python
from rcnn import batch_inference_with_results

results = batch_inference_with_results(
    config, 
    checkpoint_path, 
    image_dir,
    output_json="results.json"
)
```

### Complete Workflow Demo

Run the full pipeline:

```bash
python complete_workflow_example.py
```

This will:
1. Check prerequisites
2. Train a demo model
3. Evaluate it
4. Run inference
5. Show results

## Helpful Resources

### Documentation
- **README.md** - Full documentation
- **QUICKSTART.md** - Quick reference
- **PROJECT_SUMMARY.md** - Complete overview
- **config_example.py** - Configuration guide

### Example Scripts
- **train_example.py** - Training example
- **inference_example.py** - Inference examples
- **complete_workflow_example.py** - Full demo
- **verify_installation.py** - Check setup

## Tips for Success

1. ✅ **Start small** - Train for 5-10 epochs first
2. ✅ **Monitor progress** - Watch training curves
3. ✅ **Use GPU** - 10-20x faster than CPU
4. ✅ **Save checkpoints** - Don't lose progress
5. ✅ **Visualize results** - Always check predictions
6. ✅ **Iterate** - Experiment with hyperparameters
7. ✅ **Document** - Keep notes on experiments

## Command Reference

### Training
```bash
# Basic
python rcnn.py --mode train

# With options
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005

# Custom dataset
python rcnn.py --mode train --data-root path/to/data --data-yaml path/to/data.yaml
```

### Testing
```bash
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
```

### Inference
```bash
# Single image
python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image img.jpg

# Batch
python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image-dir images/

# Custom threshold
python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image img.jpg --conf-threshold 0.7
```

## Getting Help

If you encounter issues:

1. **Check error message** - Often tells you what's wrong
2. **Review this guide** - Solutions for common problems
3. **Run verification** - `python verify_installation.py`
4. **Check training curves** - Diagnose training issues
5. **Try smaller batch size** - If memory errors
6. **Use CPU temporarily** - If GPU issues

---

**Ready to start?**

```bash
# Verify setup
python verify_installation.py

# First training run
python rcnn.py --mode train --epochs 5 --batch-size 2

# Check results
ls checkpoints/
ls results/
```

**Happy training! 🚀**

