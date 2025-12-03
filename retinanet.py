"""
RetinaNet: Focal Loss for Dense Object Detection
================================================
A PyTorch implementation of RetinaNet for urban issue detection.
Wraps the torchvision implementation and reuses the training loop from rcnn.py.
"""

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from rcnn import Config, train as rcnn_train, evaluate as rcnn_evaluate

def get_model(num_classes: int, pretrained: bool = True):
    """
    Create a RetinaNet model with ResNet50-FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: RetinaNet model
    """
    if pretrained:
        # Load model with ImageNet pretrained backbone
        # We use weights_backbone instead of weights to allow changing num_classes easily
        model = retinanet_resnet50_fpn(
            weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            num_classes=num_classes
        )
    else:
        model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
        
    return model

from rcnn import test as rcnn_test, inference as rcnn_inference
from typing import Optional

def train(config: Config):
    """Train RetinaNet model."""
    print("Training RetinaNet model...")
    config.model_name = 'retinanet'
    return rcnn_train(config, model_builder=get_model)

def evaluate(model, data_loader, device, config):
    """Evaluate RetinaNet model."""
    return rcnn_evaluate(model, data_loader, device, config)

def test(config: Config, checkpoint_path: str):
    """Test RetinaNet model."""
    print("Testing RetinaNet model...")
    return rcnn_test(config, checkpoint_path, model_builder=get_model)

def inference(config: Config, checkpoint_path: str, image_path: str, save_path: Optional[str] = None):
    """Run inference with RetinaNet model."""
    print("Running inference with RetinaNet model...")
    return rcnn_inference(config, checkpoint_path, image_path, save_path, model_builder=get_model)
