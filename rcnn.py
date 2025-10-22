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
    
    def __init__(self):
        # Dataset paths
        self.data_root = "data/DamagedRoadSigns/DamagedRoadSigns"
        self.data_yaml = "data/DamagedRoadSigns/DamagedRoadSigns/data.yaml"
        
        # Model hyperparameters
        self.num_classes = 3  # background + 2 classes (Damage, Healthy)
        self.backbone = "resnet50"  # or resnet101
        self.pretrained = True
        
        # Training hyperparameters
        self.batch_size = 4
        self.num_epochs = 50
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.lr_scheduler_step_size = 10
        self.lr_scheduler_gamma = 0.1
        
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
        
        # Paths
        self.checkpoint_dir = "checkpoints"
        self.output_dir = "outputs"
        self.results_dir = "results"
        
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
    
    def __str__(self):
        """String representation of configuration."""
        return json.dumps(self.__dict__, default=str, indent=2)


# ============================================================================
# Dataset Loader for YOLO Format
# ============================================================================

class YOLODataset(Dataset):
    """
    Custom Dataset for loading YOLO format annotations.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1].
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transforms=None):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'valid', or 'test'
            transforms: Optional transforms to apply
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        self.image_dir = self.root_dir / split / 'images'
        self.label_dir = self.root_dir / split / 'labels'
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')) +
                                 list(self.image_dir.glob('*.jpeg')))
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            image: PIL Image
            target: dict with keys 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        # Load corresponding label file
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)
                    
                    # Validate class_id is in valid range [0, 1] for 2-class dataset
                    # Skip invalid labels to prevent CUDA assertion errors
                    if class_id < 0 or class_id >= 2:
                        continue
                    
                    # Convert YOLO format to [x_min, y_min, x_max, y_max]
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id + 1)  # +1 because 0 is background
        
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
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for i, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Statistics
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_classifier'].item()
        running_loss_box_reg += loss_dict['loss_box_reg'].item()
        running_loss_objectness += loss_dict['loss_objectness'].item()
        running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # Update progress bar
        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'cls': f'{running_loss_classifier / (i + 1):.4f}',
                'box': f'{running_loss_box_reg / (i + 1):.4f}'
            })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(data_loader)
    epoch_stats = {
        'total_loss': epoch_loss,
        'loss_classifier': running_loss_classifier / len(data_loader),
        'loss_box_reg': running_loss_box_reg / len(data_loader),
        'loss_objectness': running_loss_objectness / len(data_loader),
        'loss_rpn_box_reg': running_loss_rpn_box_reg / len(data_loader)
    }
    
    return epoch_stats


def train(config: Config):
    """Main training function."""
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"\nConfiguration:\n{config}\n")
    
    # Load class names from data.yaml
    with open(config.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = ['background'] + data_config['names']
    print(f"Classes: {class_names}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = YOLODataset(config.data_root, split='train')
    valid_dataset = YOLODataset(config.data_root, split='valid')
    
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
    model = get_model(config.num_classes, pretrained=config.pretrained)
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
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"  Classifier Loss: {epoch_stats['loss_classifier']:.4f}")
        print(f"  Box Reg Loss: {epoch_stats['loss_box_reg']:.4f}")
        print(f"  Objectness Loss: {epoch_stats['loss_objectness']:.4f}")
        print(f"  RPN Box Reg Loss: {epoch_stats['loss_rpn_box_reg']:.4f}")
        
        epoch_stats['epoch'] = epoch
        epoch_stats['lr'] = current_lr
        training_history.append(epoch_stats)
        
        # Save checkpoint
        if epoch % config.save_freq == 0 or epoch == config.num_epochs:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f'fasterrcnn_epoch_{epoch}.pth'
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
                best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
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
    history_path = os.path.join(config.results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Plot training curves
    plot_training_curves(training_history, config.results_dir)
    
    print("\nTraining completed!")
    return model


def plot_training_curves(history: List[Dict], save_dir: str):
    """Plot training curves."""
    epochs = [h['epoch'] for h in history]
    total_loss = [h['total_loss'] for h in history]
    classifier_loss = [h['loss_classifier'] for h in history]
    box_reg_loss = [h['loss_box_reg'] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, total_loss, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss vs Epoch')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, classifier_loss, 'r-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Classifier Loss')
    axes[1].set_title('Classifier Loss vs Epoch')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, box_reg_loss, 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Box Regression Loss')
    axes[2].set_title('Box Regression Loss vs Epoch')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
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


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    """Evaluate model on test set."""
    model.eval()
    
    print("=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    
    all_predictions = []
    all_ground_truths = []
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        
        # Get predictions
        predictions = model(images)
        
        # Move to CPU
        predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(targets)
    
    # Calculate mAP
    results = calculate_map(all_predictions, all_ground_truths, config.conf_threshold)
    
    print("\nEvaluation Results:")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Total Predictions: {results['total_predictions']}")
    print(f"  Total Ground Truths: {results['total_ground_truths']}")
    
    # Save results
    results_path = os.path.join(config.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def calculate_map(predictions, ground_truths, conf_threshold=0.5, iou_threshold=0.5):
    """Calculate mean Average Precision."""
    true_positives = 0
    false_positives = 0
    total_ground_truths = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Filter by confidence
        mask = pred_scores > conf_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        total_ground_truths += len(gt_boxes)
        
        matched_gt = set()
        
        # Sort predictions by score (descending)
        if len(pred_scores) > 0:
            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            
            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in matched_gt:
                        continue
                    
                    if pred_label == gt_label:
                        iou = compute_iou(pred_box.numpy(), gt_box.numpy())
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    
    # Simplified mAP calculation
    mAP = (precision + recall) / 2 if (precision + recall) > 0 else 0
    
    return {
        'mAP@0.5': mAP,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'total_ground_truths': total_ground_truths,
        'total_predictions': true_positives + false_positives
    }


def test(config: Config, checkpoint_path: str):
    """Test the model."""
    print("=" * 80)
    print("Testing Model")
    print("=" * 80)
    
    # Load class names
    with open(config.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = ['background'] + data_config['names']
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = YOLODataset(config.data_root, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes, pretrained=False)
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
def inference(config: Config, checkpoint_path: str, image_path: str, save_path: Optional[str] = None):
    """Run inference on a single image."""
    print("=" * 80)
    print("Running Inference")
    print("=" * 80)
    
    # Load class names
    with open(config.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = ['background'] + data_config['names']
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes, pretrained=False)
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
  # Train from scratch
  python rcnn.py --mode train --epochs 50 --batch-size 4 --lr 0.005
  
  # Test on test set
  python rcnn.py --mode test --checkpoint checkpoints/best_model.pth
  
  # Run inference on single image
  python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
  
  # Run inference on directory
  python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image-dir path/to/images/
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
    
    # Output directories
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
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

