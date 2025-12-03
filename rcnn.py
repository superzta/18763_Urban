"""
RUIDR: Real-Time Urban Issue Detection using Faster R-CNN
===========================================================
A PyTorch implementation of Faster R-CNN for urban issue detection.

Author: 18-794 Course Project Team
Dataset: Urban Issues Dataset (YOLO format)
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
from datetime import datetime


# ============================================================================
# Configuration Management
# ============================================================================

class Config:
    """Configuration class for managing all hyperparameters and settings."""
    
    # Map urban issue class ID to dataset folder
    URBAN_ISSUE_DATASETS = {
        0: ("Potholes and RoadCracks", "Damaged Road issues"),
        1: ("Potholes and RoadCracks", "Pothole Issues"),
        2: ("IllegalParking", "Illegal Parking Issues"),
        3: ("DamagedRoadSigns", "Broken Road Sign Issues"),
        4: ("FallenTrees", "Fallen trees"),
        5: ("Garbage", "Littering/Garbage on Public Places"),
        6: ("Graffitti", "Vandalism Issues"),
        7: ("DeadAnimalsPollution", "Dead Animal Pollution"),
        8: ("Damaged concrete structures", "Damaged concrete structures"),
        9: ("DamagedElectricalPoles", "Damaged Electric wires and poles")
    }
    
    def __init__(self, urban_issue_classes=[3]):
        """
        Initialize configuration.
        
        Args:
            urban_issue_classes: List of urban issue classes to train on (0-9)
                Default: [3] (Broken Road Sign Issues)
                Can be multiple: [0, 1, 3] for multiple issue types
        """
        # Validate classes
        for cls_id in urban_issue_classes:
            if cls_id not in self.URBAN_ISSUE_DATASETS:
                raise ValueError(f"Invalid class {cls_id}. Must be 0-9.")
        
        self.urban_issue_classes = sorted(urban_issue_classes)
        self.main_config_yaml = "data/config.yaml"
        
        # Model hyperparameters - will be set based on selected classes
        self.num_classes = len(self.urban_issue_classes) + 1  # +1 for background
        self.backbone = "resnet50"
        self.pretrained = True
        
        # Training hyperparameters
        self.batch_size = 4
        self.num_epochs = 50
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.lr_scheduler_step_size = 10
        self.lr_scheduler_gamma = 0.1
        self.warmup_epochs = 5   
        
        # Data augmentation
        self.horizontal_flip_prob = 0.5
        self.min_size = 640
        self.max_size = 640
        
        # Training settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.print_freq = 50
        self.save_freq = 5  # Save checkpoint every N epochs
        
        # Inference settings
        self.conf_threshold = 0.5  # Confidence threshold for detections
        self.nms_threshold = 0.3   # NMS IoU threshold
        self.use_proxy_map = False # Use proxy mAP (P+R)/2 instead of exact AP
        
        # Paths
        self.checkpoint_dir = "checkpoints"
        self.output_dir = "outputs"
        self.results_dir = "results"
        
        self.model_name = "fasterrcnn"  # Default model name
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments."""
        if args.data_root:
            self.data_root = args.data_root
        if args.data_yaml:
            self.data_yaml = args.data_yaml
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.epochs:
            self.num_epochs = args.epochs
        if args.lr:
            self.learning_rate = args.lr
        if args.conf_threshold:
            self.conf_threshold = args.conf_threshold
        if args.checkpoint_dir:
            self.checkpoint_dir = args.checkpoint_dir
        if args.output_dir:
            self.output_dir = args.output_dir
        if args.use_proxy_map:
            self.use_proxy_map = True
    
    def __str__(self):
        """String representation of configuration."""
        return json.dumps(self.__dict__, default=str, indent=2)


# ============================================================================
# Dataset Loader for YOLO Format
# ============================================================================

class MultiClassUrbanDataset(Dataset):
    """
    Combined dataset loader for multiple urban issue classes.
    Handles loading from multiple subdatasets and mapping to global class IDs.
    """
    
    def __init__(self, config: 'Config', split: str = 'train', transforms=None):
        """
        Args:
            config: Config object with urban_issue_classes specified
            split: 'train', 'valid', or 'test'
            transforms: Optional transforms to apply
        """
        self.config = config
        self.split = split
        self.transforms = transforms
        
        # Load main config to get class names
        with open(config.main_config_yaml, 'r') as f:
            main_config = yaml.safe_load(f)
        self.global_class_names = main_config['names']
        
        # Build dataset: collect all images from selected classes
        self.samples = []  # List of (image_path, label_path, global_class_id)
        
        for global_class_id in config.urban_issue_classes:
            folder_name, class_name = config.URBAN_ISSUE_DATASETS[global_class_id]
            dataset_root = Path(f"data/{folder_name}/{folder_name}")
            
            image_dir = dataset_root / split / 'images'
            label_dir = dataset_root / split / 'labels'
            
            if not image_dir.exists():
                print(f"Warning: {image_dir} not found, skipping class {global_class_id}")
                continue
            
            # Get all images for this class
            image_files = sorted(list(image_dir.glob('*.jpg')) + 
                                list(image_dir.glob('*.png')) +
                                list(image_dir.glob('*.jpeg')))
            
            for img_path in image_files:
                label_path = label_dir / (img_path.stem + '.txt')
                self.samples.append((img_path, label_path, global_class_id))
            
            print(f"  Class {global_class_id} ({class_name}): {len(image_files)} images")
        
        print(f"Loaded total {len(self.samples)} images from {split} split")
        
        # Create mapping from global class ID to model output index
        # Model outputs: 0=background, 1=first selected class, 2=second selected class, etc.
        self.global_to_model_idx = {global_cls: i+1 for i, global_cls in enumerate(config.urban_issue_classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            image: PIL Image
            target: dict with keys 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        img_path, label_path, expected_global_class = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    local_class_id, x_center, y_center, width, height = map(float, parts)
                    local_class_id = int(local_class_id)
                    
                    # The label file contains local class IDs specific to this subdataset
                    # We need to map them to the global class ID
                    # For most subdatasets, all labels should map to the same global class
                    # (e.g., all labels in DamagedRoadSigns folder are class 3)
                    
                    # Use the expected global class for this image
                    global_class_id = expected_global_class
                    
                    # Map global class ID to model output index
                    if global_class_id not in self.global_to_model_idx:
                        continue  # Skip if this class wasn't selected for training
                    
                    model_class_idx = self.global_to_model_idx[global_class_id]
                    
                    # Convert YOLO format to [x_min, y_min, x_max, y_max]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    # Validate bounding box - must have positive width and height
                    # Skip invalid boxes (zero or negative dimensions)
                    if x_max <= x_min or y_max <= y_min:
                        continue  # Skip this invalid box
                    
                    # Also ensure box is within image boundaries
                    x_min = max(0, min(x_min, img_width))
                    y_min = max(0, min(y_min, img_height))
                    x_max = max(0, min(x_max, img_width))
                    y_max = max(0, min(y_max, img_height))
                    
                    # Final check after clipping
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(model_class_idx)
        
        # Convert to tensors
        if len(boxes) == 0:
            # If no boxes, create dummy box
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        # Calculate area
        if len(boxes) > 0:
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            target['area'] = torch.zeros((0,), dtype=torch.float32)
        
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            # Convert image to tensor
            image = F.to_tensor(image)
        
        return image, target


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))


# ============================================================================
# Model Definition
# ============================================================================

def get_model(num_classes: int, pretrained: bool = True):
    """
    Create a Faster R-CNN model with ResNet50-FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: Faster R-CNN model
    """
    # Load pretrained model
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_losses = {}  # Dynamic dictionary for all loss components
    
    skipped_batches = 0
    successful_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for i, (images, targets) in enumerate(pbar):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Filter out targets with no valid boxes
            valid_indices = [idx for idx, t in enumerate(targets) if len(t['boxes']) > 0]
            
            if len(valid_indices) == 0:
                skipped_batches += 1
                continue  # Skip this batch if no valid targets
            
            # Keep only valid samples
            images = [images[idx] for idx in valid_indices]
            targets = [targets[idx] for idx in valid_indices]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Statistics
            running_loss += losses.item()
            
            # Dynamically track all losses
            for k, v in loss_dict.items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v.item()
            
            successful_batches += 1
            
            # Update progress bar
            if (successful_batches + 1) % print_freq == 0:
                avg_loss = running_loss / successful_batches if successful_batches > 0 else 0
                
                # Create postfix dict with main loss and up to 2 components
                postfix = {'loss': f'{avg_loss:.4f}'}
                
                # Add first 2 components to pbar
                count = 0
                for k in sorted(running_losses.keys()):
                    if count >= 2: break
                    postfix[k] = f'{running_losses[k] / successful_batches:.4f}'
                    count += 1
                
                postfix['skipped'] = skipped_batches
                pbar.set_postfix(postfix)
        
        except Exception as e:
            # Log error and continue training
            skipped_batches += 1
            if skipped_batches <= 5:  # Only print first 5 errors to avoid spam
                print(f"\nWarning: Skipped batch {i} due to error: {str(e)[:100]}")
            continue
    
    # Calculate epoch statistics
    if successful_batches > 0:
        epoch_loss = running_loss / successful_batches
        epoch_stats = {
            'total_loss': epoch_loss,
            'skipped_batches': skipped_batches,
            'successful_batches': successful_batches
        }
        # Add all individual losses
        for k, v in running_losses.items():
            epoch_stats[k] = v / successful_batches
    else:
        # No successful batches - return zero losses
        epoch_stats = {
            'total_loss': 0.0,
            'skipped_batches': skipped_batches,
            'successful_batches': 0
        }
    
    if skipped_batches > 0:
        print(f"\nEpoch summary: {successful_batches} successful batches, {skipped_batches} skipped batches")
    
    return epoch_stats


def train(config: Config, model_builder=None):
    """Main training function."""
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Build class names list: background + selected classes
    selected_class_names = ['background']
    for cls_id in config.urban_issue_classes:
        selected_class_names.append(main_config['names'][cls_id])
    
    print(f"Training on {len(config.urban_issue_classes)} urban issue class(es):")
    for cls_id in config.urban_issue_classes:
        print(f"  Class {cls_id}: {main_config['names'][cls_id]}")
    print(f"\nModel classes: {selected_class_names}")
    print(f"Number of classes (including background): {config.num_classes}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MultiClassUrbanDataset(config, split='train')
    valid_dataset = MultiClassUrbanDataset(config, split='valid')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model on device: {config.device}")
    if model_builder is None:
        model = get_model(config.num_classes, pretrained=config.pretrained)
    else:
        model = model_builder(config.num_classes, pretrained=config.pretrained)
    model.to(config.device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_scheduler_step_size,
        gamma=config.lr_scheduler_gamma
    )
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    training_history = []
    best_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        epoch_stats = train_one_epoch(
            model, optimizer, train_loader, config.device, epoch, config.print_freq
        )
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log statistics
        # Log statistics
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Total Loss: {epoch_stats['total_loss']:.4f}")
        
        # Log all individual losses
        for k, v in epoch_stats.items():
            if k not in ['total_loss', 'epoch', 'lr', 'skipped_batches', 'successful_batches']:
                print(f"  {k}: {v:.4f}")
        if epoch_stats.get('skipped_batches', 0) > 0:
            print(f"  ⚠ Skipped batches: {epoch_stats['skipped_batches']} (invalid data)")
            print(f"  ✓ Successful batches: {epoch_stats['successful_batches']}")
        
        epoch_stats['epoch'] = epoch
        epoch_stats['lr'] = current_lr
        training_history.append(epoch_stats)
        
        # Save checkpoint
        # Save checkpoint
        if epoch % config.save_freq == 0 or epoch == config.num_epochs:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f'{config.model_name}_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': epoch_stats['total_loss'],
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if epoch_stats['total_loss'] < best_loss:
                best_loss = epoch_stats['total_loss']
                best_path = os.path.join(config.checkpoint_dir, f'{config.model_name}_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config.__dict__
                }, best_path)
                print(f"  Saved best model: {best_path}")
        
        print()
    
    # Save training history
    history_path = os.path.join(config.results_dir, f'{config.model_name}_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Plot training curves
    plot_training_curves(training_history, config.results_dir, config.model_name)
    
    print("\nTraining completed!")
    return model


def plot_training_curves(history: List[Dict], save_dir: str, model_name: str = 'model'):
    """Plot training curves."""
    if not history:
        return
        
    epochs = [h['epoch'] for h in history]
    total_loss = [h['total_loss'] for h in history]
    
    # Identify all loss keys present in history
    loss_keys = set()
    for h in history:
        for k in h.keys():
            if k not in ['total_loss', 'epoch', 'lr', 'skipped_batches', 'successful_batches']:
                loss_keys.add(k)
    
    loss_keys = sorted(list(loss_keys))
    num_plots = 1 + len(loss_keys)
    
    # Calculate grid size
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    
    # Plot Total Loss
    axes[0].plot(epochs, total_loss, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss vs Epoch')
    axes[0].grid(True, alpha=0.3)
    
    # Plot individual losses
    for i, key in enumerate(loss_keys):
        values = [h.get(key, 0.0) for h in history]
        ax = axes[i+1]
        ax.plot(epochs, values, 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(f'{key} vs Epoch')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def create_evaluation_visualizations(images, predictions, ground_truths, config, num_samples=10):
    """Create visualizations comparing predictions vs ground truth."""
    
    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Build class names list for selected classes
    class_names = ['background']
    for cls_id in config.urban_issue_classes:
        class_names.append(main_config['names'][cls_id])
    
    # Select samples with ground truth boxes
    samples_to_viz = []
    for i, gt in enumerate(ground_truths):
        if len(gt['boxes']) > 0:
            samples_to_viz.append(i)
        if len(samples_to_viz) >= num_samples:
            break
    
    if not samples_to_viz:
        print("No samples with ground truth boxes found for visualization")
        return
    
    # Create comparison visualizations
    for idx, sample_idx in enumerate(samples_to_viz):
        img_tensor = images[sample_idx]
        pred = predictions[sample_idx]
        gt = ground_truths[sample_idx]
        
        # Convert tensor to PIL Image
        img = F.to_pil_image(img_tensor)
        
        # Create figure with 3 subplots: Ground Truth, Predictions, Both
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Ground Truth
        axes[0].imshow(img)
        axes[0].set_title(f'Ground Truth ({len(gt["boxes"])} boxes)', fontsize=12, fontweight='bold')
        for box, label in zip(gt['boxes'], gt['labels']):
            x_min, y_min, x_max, y_max = box.numpy()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x_min, y_min - 5,
                f"GT: {class_names[label]}",
                color='white', fontsize=9,
                bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=2)
            )
        axes[0].axis('off')
        
        # Predictions (filtered by confidence)
        mask = pred['scores'] > config.conf_threshold
        pred_boxes = pred['boxes'][mask]
        pred_labels = pred['labels'][mask]
        pred_scores = pred['scores'][mask]
        
        axes[1].imshow(img)
        axes[1].set_title(f'Predictions ({len(pred_boxes)} boxes, conf>{config.conf_threshold})', 
                         fontsize=12, fontweight='bold')
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x_min, y_min, x_max, y_max = box.numpy()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x_min, y_min - 5,
                f"Pred: {class_names[label]} ({score:.2f})",
                color='white', fontsize=9,
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=2)
            )
        axes[1].axis('off')
        
        # Both overlaid
        axes[2].imshow(img)
        axes[2].set_title('Ground Truth (Green) vs Predictions (Red)', fontsize=12, fontweight='bold')
        
        # Draw ground truth in green
        for box, label in zip(gt['boxes'], gt['labels']):
            x_min, y_min, x_max, y_max = box.numpy()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
            )
            axes[2].add_patch(rect)
        
        # Draw predictions in red
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x_min, y_min, x_max, y_max = box.numpy()
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(config.results_dir, f'eval_comparison_{idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(samples_to_viz)} comparison visualizations to {config.results_dir}/eval_comparison_*.png")


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    """Evaluate model on test set."""
    model.eval()
    
    print("=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    
    all_predictions = []
    all_ground_truths = []
    all_images = []
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for images, targets in pbar:
        # Store images for visualization
        all_images.extend([img.cpu() for img in images])
        
        images = list(img.to(device) for img in images)
        
        # Get predictions
        predictions = model(images)
        
        # Move to CPU
        predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(targets)
    
    # Calculate mAP
    results = calculate_map(
        all_predictions, 
        all_ground_truths, 
        config.conf_threshold,
        use_proxy=config.use_proxy_map
    )
    
    print("\nEvaluation Results:")
    print(f"  mAP@0.5 ({'Proxy' if results['is_proxy'] else 'Exact'}): {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Total Predictions: {results['total_predictions']}")
    print(f"  Total Ground Truths: {results['total_ground_truths']}")
    
    # Save results
    results_path = os.path.join(config.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Create visualizations comparing predictions vs ground truth
    print("\nCreating evaluation visualizations...")
    create_evaluation_visualizations(
        all_images, all_predictions, all_ground_truths, 
        config, num_samples=10
    )
    
    return results


# def calculate_map(predictions, ground_truths, conf_threshold=0.5, iou_threshold=0.5, use_proxy=False):
#     """
#     Calculate mean Average Precision (mAP) across all classes.
    
#     Args:
#         predictions: List of prediction dicts
#         ground_truths: List of ground truth dicts
#         conf_threshold: Confidence threshold (used only for proxy mAP)
#         iou_threshold: IoU threshold for matching
#         use_proxy: If True, use (Precision+Recall)/2 as proxy for AP.
#                   If False, use exact Area Under Precision-Recall Curve (VOC style).
#     """
#     # Find all unique classes in GT and Predictions
#     all_classes = set()
#     for gt in ground_truths:
#         all_classes.update(gt['labels'].tolist())
#     for pred in predictions:
#         all_classes.update(pred['labels'].tolist())
    
#     if not all_classes:
#         return {
#             'mAP@0.5': 0.0,
#             'precision': 0.0,
#             'recall': 0.0,
#             'total_predictions': 0,
#             'total_ground_truths': 0,
#             'is_proxy': use_proxy
#         }

#     # Per-class metrics
#     aps = []
#     precisions = []
#     recalls = []
#     total_tps = 0
#     total_fps = 0
#     total_gts = 0
    
#     for class_id in all_classes:
#         class_id = int(class_id)
        
#         # Collect detections and GTs for this class
#         class_detections = [] # (score, is_tp)
#         class_n_gts = 0
        
#         class_tps = 0
#         class_fps = 0
        
#         for pred, gt in zip(predictions, ground_truths):
#             # Get GT boxes for this class
#             gt_mask = gt['labels'] == class_id
#             gt_boxes = gt['boxes'][gt_mask]
#             class_n_gts += len(gt_boxes)
            
#             # Get Pred boxes for this class
#             pred_mask = pred['labels'] == class_id
#             pred_boxes = pred['boxes'][pred_mask]
#             pred_scores = pred['scores'][pred_mask]
            
#             # For Exact AP, we use ALL detections (no threshold or very low threshold)
#             # For Proxy AP, we use the provided threshold
#             if use_proxy:
#                 score_mask = pred_scores > conf_threshold
#             else:
#                 score_mask = pred_scores > 0.0 # Use all for PR curve
                
#             pred_boxes = pred_boxes[score_mask]
#             pred_scores = pred_scores[score_mask]
            
#             # Match predictions to GT
#             matched_gt = set()
            
#             # Sort by score descending
#             if len(pred_scores) > 0:
#                 sorted_indices = torch.argsort(pred_scores, descending=True)
#                 pred_boxes = pred_boxes[sorted_indices]
#                 pred_scores = pred_scores[sorted_indices]
                
#                 for box, score in zip(pred_boxes, pred_scores):
#                     best_iou = 0
#                     best_gt_idx = -1
                    
#                     for gt_idx, gt_box in enumerate(gt_boxes):
#                         if gt_idx in matched_gt:
#                             continue
                        
#                         iou = compute_iou(box.numpy(), gt_box.numpy())
#                         if iou > best_iou:
#                             best_iou = iou
#                             best_gt_idx = gt_idx
                    
#                     is_tp = False
#                     if best_iou >= iou_threshold and best_gt_idx != -1:
#                         matched_gt.add(best_gt_idx)
#                         is_tp = True
#                         class_tps += 1
#                     else:
#                         class_fps += 1
                    
#                     class_detections.append((score.item(), is_tp))
        
#         # Calculate AP for this class
#         if class_n_gts == 0:
#             ap = 0.0
#             prec = 0.0
#             rec = 0.0
#         else:
#             if use_proxy:
#                 # Proxy: (P+R)/2 at specific threshold
#                 prec = class_tps / (class_tps + class_fps) if (class_tps + class_fps) > 0 else 0
#                 rec = class_tps / class_n_gts
#                 ap = (prec + rec) / 2
#             else:
#                 # Exact: Area Under PR Curve
#                 if len(class_detections) == 0:
#                     ap = 0.0
#                     prec = 0.0
#                     rec = 0.0
#                 else:
#                     # Sort by score
#                     class_detections.sort(key=lambda x: x[0], reverse=True)
                    
#                     tps = np.array([1 if x[1] else 0 for x in class_detections])
#                     fps = np.array([1 if not x[1] else 0 for x in class_detections])
                    
#                     cum_tps = np.cumsum(tps)
#                     cum_fps = np.cumsum(fps)
                    
#                     prec_curve = cum_tps / (cum_tps + cum_fps)
#                     rec_curve = cum_tps / class_n_gts
                    
#                     # Pad for AUC
#                     prec_curve = np.concatenate(([0.0], prec_curve, [0.0]))
#                     rec_curve = np.concatenate(([0.0], rec_curve, [1.0]))
                    
#                     # Monotonically decreasing precision
#                     for i in range(len(prec_curve) - 2, -1, -1):
#                         prec_curve[i] = max(prec_curve[i], prec_curve[i + 1])
                    
#                     # AUC
#                     indices = np.where(rec_curve[1:] != rec_curve[:-1])[0]
#                     ap = np.sum((rec_curve[indices + 1] - rec_curve[indices]) * prec_curve[indices + 1])
                    
#                     # For reporting, use the values at the end (or max) - simplified for summary
#                     prec = prec_curve[-2] if len(prec_curve) > 1 else 0 # Approximation for summary
#                     rec = rec_curve[-2] if len(rec_curve) > 1 else 0
        
#         aps.append(ap)
#         precisions.append(prec)
#         recalls.append(rec)
#         total_tps += class_tps
#         total_fps += class_fps
#         total_gts += class_n_gts

#     # Mean metrics
#     mAP = np.mean(aps) if aps else 0.0
#     mean_prec = np.mean(precisions) if precisions else 0.0
#     mean_rec = np.mean(recalls) if recalls else 0.0
    
#     return {
#         'mAP@0.5': float(mAP),
#         'precision': float(mean_prec),
#         'recall': float(mean_rec),
#         'total_predictions': total_tps + total_fps,
#         'total_ground_truths': total_gts,
#         'is_proxy': use_proxy
#     }

def calculate_map(predictions, ground_truths, conf_threshold=0.5, iou_threshold=0.5, use_proxy=False):
    """
    Calculate mean Average Precision (mAP) across all (non-background) classes.

    Args:
        predictions: List[dict] with keys 'boxes', 'labels', 'scores' (torch tensors)
        ground_truths: List[dict] with keys 'boxes', 'labels' (torch tensors)
        conf_threshold: Confidence threshold (used only for proxy mAP)
        iou_threshold: IoU threshold for matching
        use_proxy: If True, use (Precision+Recall)/2 per class at a single threshold.
                   If False, use exact Area Under Precision-Recall Curve (VOC style).
    """
    # Collect all class ids
    all_classes = set()
    for gt in ground_truths:
        all_classes.update(gt["labels"].tolist())
    for pred in predictions:
        all_classes.update(pred["labels"].tolist())

    all_classes = {int(c) for c in all_classes}
    # Remove background if present
    if 0 in all_classes:
        all_classes.remove(0)

    if not all_classes:
        return {
            "mAP@0.5": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "total_predictions": 0,
            "total_ground_truths": 0,
            "is_proxy": use_proxy,
        }

    aps = []
    per_class_precisions = []
    per_class_recalls = []
    total_tps = 0
    total_fps = 0
    total_gts = 0

    for class_id in all_classes:
        class_detections = []  # list of (score, is_tp)
        class_n_gts = 0
        class_tps = 0
        class_fps = 0

        for pred, gt in zip(predictions, ground_truths):
            # GT for this class
            gt_mask = (gt["labels"] == class_id)
            gt_boxes = gt["boxes"][gt_mask]
            class_n_gts += len(gt_boxes)

            # Predictions for this class
            pred_mask = (pred["labels"] == class_id)
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            # Proxy: threshold; Exact: keep all detections
            if use_proxy:
                score_mask = pred_scores > conf_threshold
            else:
                score_mask = pred_scores > 0.0

            pred_boxes = pred_boxes[score_mask]
            pred_scores = pred_scores[score_mask]

            matched_gt = set()

            if len(pred_scores) > 0:
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]

                for box, score in zip(pred_boxes, pred_scores):
                    best_iou = 0.0
                    best_gt_idx = -1

                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue

                        iou = compute_iou(box.numpy(), gt_box.numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    is_tp = False
                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        matched_gt.add(best_gt_idx)
                        is_tp = True
                        class_tps += 1
                    else:
                        class_fps += 1

                    class_detections.append((score.item(), is_tp))

        if class_n_gts == 0:
            ap = 0.0
            prec = 0.0
            rec = 0.0
        else:
            if use_proxy:
                # Single-threshold proxy
                prec = class_tps / (class_tps + class_fps) if (class_tps + class_fps) > 0 else 0.0
                rec = class_tps / class_n_gts
                ap = (prec + rec) / 2.0
            else:
                # Exact AP via PR curve
                if len(class_detections) == 0:
                    ap = 0.0
                    prec = 0.0
                    rec = 0.0
                else:
                    class_detections.sort(key=lambda x: x[0], reverse=True)
                    tps = np.array([1 if x[1] else 0 for x in class_detections], dtype=np.float32)
                    fps = np.array([1 if not x[1] else 0 for x in class_detections], dtype=np.float32)

                    cum_tps = np.cumsum(tps)
                    cum_fps = np.cumsum(fps)

                    prec_curve = cum_tps / (cum_tps + cum_fps)
                    rec_curve = cum_tps / class_n_gts

                    prec_curve = np.concatenate(([0.0], prec_curve, [0.0]))
                    rec_curve = np.concatenate(([0.0], rec_curve, [1.0]))

                    for i in range(len(prec_curve) - 2, -1, -1):
                        prec_curve[i] = max(prec_curve[i], prec_curve[i + 1])

                    indices = np.where(rec_curve[1:] != rec_curve[:-1])[0]
                    ap = float(np.sum((rec_curve[indices + 1] - rec_curve[indices]) * prec_curve[indices + 1]))

                    # For reporting: global P/R from counts
                    prec = class_tps / (class_tps + class_fps) if (class_tps + class_fps) > 0 else 0.0
                    rec = class_tps / class_n_gts

        aps.append(ap)
        per_class_precisions.append(prec)
        per_class_recalls.append(rec)
        total_tps += class_tps
        total_fps += class_fps
        total_gts += class_n_gts

    mAP = float(np.mean(aps)) if aps else 0.0
    mean_prec = float(np.mean(per_class_precisions)) if per_class_precisions else 0.0
    mean_rec = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0

    return {
        "mAP@0.5": mAP,
        "precision": mean_prec,
        "recall": mean_rec,
        "total_predictions": total_tps + total_fps,
        "total_ground_truths": total_gts,
        "is_proxy": use_proxy,
    }



def test(config: Config, checkpoint_path: str, model_builder=None):
    """Test the model."""
    print("=" * 80)
    print("Testing Model")
    print("=" * 80)
    
    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Build class names list
    class_names = ['background']
    for cls_id in config.urban_issue_classes:
        class_names.append(main_config['names'][cls_id])
    
    print(f"Testing on {len(config.urban_issue_classes)} urban issue class(es):")
    for cls_id in config.urban_issue_classes:
        print(f"  Class {cls_id}: {main_config['names'][cls_id]}")
    print()
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = MultiClassUrbanDataset(config, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    if model_builder is None:
        model = get_model(config.num_classes, pretrained=False)
    else:
        model = model_builder(config.num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    
    # Evaluate
    results = evaluate(model, test_loader, config.device, config)
    
    return results


# ============================================================================
# Inference Functions
# ============================================================================

@torch.no_grad()
def inference(config: Config, checkpoint_path: str, image_path: str, save_path: Optional[str] = None, model_builder=None):
    """Run inference on a single image."""
    print("=" * 80)
    print("Running Inference")
    print("=" * 80)
    
    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Build class names list
    class_names = ['background']
    for cls_id in config.urban_issue_classes:
        class_names.append(main_config['names'][cls_id])
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    if model_builder is None:
        model = get_model(config.num_classes, pretrained=False)
    else:
        model = model_builder(config.num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0).to(config.device)
    
    # Run inference
    print("Running inference...")
    predictions = model(image_tensor)[0]
    
    # Filter by confidence
    mask = predictions['scores'] > config.conf_threshold
    boxes = predictions['boxes'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = class_names[label]
        print(f"  {i+1}. {class_name}: {score:.3f} - Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Visualize
    visualize_predictions(image, boxes, labels, scores, class_names, save_path)
    
    return boxes, labels, scores


def visualize_predictions(image, boxes, labels, scores, class_names, save_path=None):
    """Visualize predictions on image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define colors for each class
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        color = colors[label % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f"{class_names[label]}: {score:.2f}"
        ax.text(
            x_min, y_min - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Faster R-CNN for Urban Issue Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on single class (default: class 3)
  python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
  
  # Train on multiple classes
  python rcnn.py --mode train --classes 0,1,3 --epochs 50 --batch-size 4
  
  # Test on specific classes
  python rcnn.py --mode test --classes 3 --checkpoint checkpoints/best_model.pth
  python rcnn.py --mode test --classes 0,1,3 --checkpoint checkpoints/best_model.pth
  
  # Run inference on single image
  python rcnn.py --mode inference --classes 3 --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
  
  # Run inference on directory
  python rcnn.py --mode inference --classes 0,1,3 --checkpoint checkpoints/best_model.pth --image-dir path/to/images/
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'test', 'inference'],
                       help='Mode: train, test, or inference')
    
    # Dataset paths
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory of dataset')
    parser.add_argument('--data-yaml', type=str, default=None,
                       help='Path to data.yaml file')
    parser.add_argument('--classes', type=str, default=None,
                       help='Comma-separated class IDs (0-9) to train/test on, e.g., "3" or "0,1,3" (default: 3)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # Inference parameters
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory of images for batch inference')
    parser.add_argument('--conf-threshold', type=float, default=None,
                       help='Confidence threshold for detections')
    parser.add_argument('--use-proxy-map', action='store_true',
                       help='Use proxy mAP calculation (P+R)/2 instead of exact AP')
    
    # Output directories
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Parse class IDs if provided
    urban_issue_classes = [3]  # Default
    if args.classes:
        try:
            urban_issue_classes = [int(x.strip()) for x in args.classes.split(',')]
        except:
            print(f"Warning: Invalid --classes format '{args.classes}', using default [3]")
            urban_issue_classes = [3]
    
    # Create config with specified classes
    config = Config(urban_issue_classes=urban_issue_classes)
    config.update_from_args(args)
    
    # Execute based on mode
    if args.mode == 'train':
        train(config)
    
    elif args.mode == 'test':
        if not args.checkpoint:
            print("Error: --checkpoint is required for test mode")
            return
        test(config, args.checkpoint)
    
    elif args.mode == 'inference':
        if not args.checkpoint:
            print("Error: --checkpoint is required for inference mode")
            return
        
        if args.image:
            # Single image inference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(config.output_dir, f'inference_{timestamp}.jpg')
            inference(config, args.checkpoint, args.image, save_path)
        
        elif args.image_dir:
            # Batch inference
            image_dir = Path(args.image_dir)
            image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
            
            print(f"\nFound {len(image_files)} images in {args.image_dir}")
            
            for img_path in tqdm(image_files, desc="Processing images"):
                save_path = os.path.join(
                    config.output_dir, 
                    f'inference_{img_path.stem}.jpg'
                )
                inference(config, args.checkpoint, str(img_path), save_path)
        
        else:
            print("Error: Either --image or --image-dir is required for inference mode")


if __name__ == '__main__':
    main()

