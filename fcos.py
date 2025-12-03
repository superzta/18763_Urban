"""
FCOS: Fully Convolutional One-Stage Object Detection
====================================================
A PyTorch implementation of FCOS for urban issue detection.
Wraps the torchvision implementation and reuses the training loop from rcnn.py.
"""

import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from rcnn import Config, train as rcnn_train, evaluate as rcnn_evaluate

def get_model(num_classes: int, pretrained: bool = True):
    """
    Create a FCOS model with ResNet50-FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: FCOS model
    """
    # Load pretrained model
    if pretrained:
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(weights=weights)
    else:
        model = fcos_resnet50_fpn(weights=None)
    
    # Replace the head
    # FCOS head is different from Faster R-CNN
    # We need to create a new head with the correct number of classes
    # The torchvision implementation doesn't have a simple 'box_predictor' replacement like Faster R-CNN
    # But we can reconstruct the model with num_classes if we are not strictly loading all weights,
    # OR we can modify the head.
    
    # However, fcos_resnet50_fpn allows passing num_classes directly if we don't use weights,
    # or if we do use weights, we might need to replace the head.
    # Let's see how torchvision handles it. 
    # Usually for transfer learning with torchvision detection models:
    # model = fcos_resnet50_fpn(weights=weights)
    # model.head = ...
    
    # Actually, the easiest way for FCOS in torchvision is often to just create a new model 
    # with the right num_classes and load the backbone weights manually if needed, 
    # OR use the `num_classes` argument which might reset the head.
    
    # Let's try the standard approach for torchvision models:
    # If we pass weights, num_classes is ignored or must match.
    # So we load with weights (COCO 91 classes), then replace the head.
    
    # FCOS head replacement:
    # The head is `model.head`. It is an FCOSHead.
    # We can create a new FCOSHead or just use the factory function with num_classes
    # and weights_backbone instead of weights.
    
    if pretrained:
        # Load model with COCO weights (91 classes)
        # We can't easily replace just the last layer of FCOSHead because it's a complex module.
        # Better approach: Load backbone weights, then create model with correct num_classes.
        
        # Option 2: Use weights_backbone
        # This is supported in newer torchvision versions for retinanet/fcos
        # But let's check if we can just do:
        model = fcos_resnet50_fpn(weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
    else:
        model = fcos_resnet50_fpn(weights=None, num_classes=num_classes)
        
    return model

import torchvision
from rcnn import test as rcnn_test, inference as rcnn_inference
from typing import Optional

def train(config: Config):
    """Train FCOS model."""
    print("Training FCOS model...")
    config.model_name = 'fcos'
    return rcnn_train(config, model_builder=get_model)

def evaluate(model, data_loader, device, config):
    """Evaluate FCOS model."""
    return rcnn_evaluate(model, data_loader, device, config)

def test(config: Config, checkpoint_path: str):
    """Test FCOS model."""
    print("Testing FCOS model...")
    return rcnn_test(config, checkpoint_path, model_builder=get_model)

def inference(config: Config, checkpoint_path: str, image_path: str, save_path: Optional[str] = None):
    """Run inference with FCOS model."""
    print("Running inference with FCOS model...")
    return rcnn_inference(config, checkpoint_path, image_path, save_path, model_builder=get_model)
