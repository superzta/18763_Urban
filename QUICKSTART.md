# Quick Start Guide

Get started with training your Faster R-CNN model in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Your Data

Make sure your data is organized like this:

```
data/DamagedRoadSigns/DamagedRoadSigns/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Step 3: Train Your Model

**Quick training (10 epochs for testing):**
```bash
python rcnn.py --mode train --epochs 10 --batch-size 2
```

**Full training (recommended):**
```bash
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
```

**Training with custom settings:**
```bash
python rcnn.py --mode train \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --data-root data/DamagedRoadSigns/DamagedRoadSigns
```

## Step 4: Test Your Model

```bash
python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
```

## Step 5: Run Inference

**Single image:**
```bash
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image data/DamagedRoadSigns/DamagedRoadSigns/test/images/sample.jpg
```

**Batch inference:**
```bash
python rcnn.py --mode inference \
    --checkpoint checkpoints/best_model.pth \
    --image-dir data/DamagedRoadSigns/DamagedRoadSigns/test/images/
```

## Common Issues

### 1. CUDA Out of Memory
**Solution**: Reduce batch size
```bash
python rcnn.py --mode train --epochs 50 --batch-size 1
```

### 2. No GPU Available
**Solution**: The code will automatically use CPU. To use GPU, install CUDA-enabled PyTorch:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Dataset Not Found
**Solution**: Update the data path:
```bash
python rcnn.py --mode train \
    --data-root path/to/your/dataset \
    --data-yaml path/to/your/data.yaml
```

## Output Files

After training, you'll find:
- `checkpoints/best_model.pth` - Your trained model
- `checkpoints/fasterrcnn_epoch_*.pth` - Checkpoints every 5 epochs
- `results/training_curves.png` - Training loss curves
- `results/training_history.json` - Detailed training logs
- `outputs/inference_*.jpg` - Inference results with bounding boxes

## Next Steps

1. **Monitor training**: Check `results/training_curves.png` to see if your model is learning
2. **Evaluate performance**: Use test mode to get mAP, precision, and recall metrics
3. **Adjust hyperparameters**: If results are not good, try:
   - Increasing epochs (100+)
   - Adjusting learning rate (0.001 - 0.01)
   - Changing batch size (2, 4, 8, 16)
4. **Run inference**: Test on your own images

## Hyperparameter Tuning Tips

| If... | Try... |
|-------|--------|
| Training loss is not decreasing | Lower learning rate (0.001 or 0.002) |
| Model is overfitting | Add more data augmentation, reduce epochs |
| Training is too slow | Increase batch size, reduce num_workers |
| CUDA OOM | Reduce batch size to 1 or 2 |
| Low mAP | Train longer (100+ epochs), check data quality |

## Example Training Commands for Different Scenarios

### Fast Experimentation (5-10 minutes)
```bash
python rcnn.py --mode train --epochs 5 --batch-size 2
```

### Standard Training (1-2 hours)
```bash
python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
```

### High-Quality Training (4-8 hours)
```bash
python rcnn.py --mode train --epochs 100 --batch-size 8 --lr 0.005
```

### Fine-tuning from Checkpoint
```bash
# Train -> interrupt -> resume (manually adjust code to load checkpoint)
python rcnn.py --mode train --epochs 100 --batch-size 4
```

## Getting Help

If you encounter any issues:
1. Check the error message carefully
2. Verify your dataset structure matches the expected format
3. Make sure all dependencies are installed
4. Try reducing batch size if you get memory errors
5. Check GPU availability with `torch.cuda.is_available()` in Python

Happy training! 🚀

