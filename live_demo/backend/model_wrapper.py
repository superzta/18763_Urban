import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path to import rcnn, fcos, retinanet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rcnn import Config, get_model as get_rcnn_model
from fcos import get_model as get_fcos_model
from retinanet import get_model as get_retinanet_model
from torchvision.transforms import functional as F
from .severity_classifier import SeverityClassifier

class ModelWrapper:
    def __init__(self, model_type='rcnn', checkpoint_path=None, classes=[3], conf_threshold=0.5, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config(urban_issue_classes=classes)
        self.config.conf_threshold = conf_threshold
        self.model_type = model_type
        
        # Create inverse mapping: model output idx -> global class ID
        # Training uses: global_to_model_idx = {global_cls: i+1 for i, global_cls in enumerate(classes)}
        # So for classes=[0,1,3]: model outputs 1→class 0, 2→class 1, 3→class 3
        self.model_idx_to_global = {i+1: global_cls for i, global_cls in enumerate(classes)}
        
        print(f"Loading {model_type} model from {checkpoint_path}...")
        
        if model_type == 'rcnn':
            self.model = get_rcnn_model(self.config.num_classes, pretrained=False)
        elif model_type == 'fcos':
            self.model = get_fcos_model(self.config.num_classes, pretrained=False)
        elif model_type == 'retinanet':
            self.model = get_retinanet_model(self.config.num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Optimize for inference
        if self.device.type == 'cuda':
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)
            print(f"Using GPU: {gpu_name} (Device Index: {device_index})")
            # Enable cuDNN autotuner for faster convolutions
            torch.backends.cudnn.benchmark = True
            # Use TensorFloat32 on Ampere GPUs for faster matmul
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("Using CPU for inference")
        
        # Initialize severity classifier
        checkpoint_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../clustering/clustering_checkpoints'))
        self.severity_classifier = SeverityClassifier(checkpoint_base_dir=checkpoint_base, device=self.device)
        print(f"[ModelWrapper] Severity classifier initialized")
        
    def predict(self, image):
        """
        Run inference on a single image with severity classification.
        Args:
            image: PIL Image or numpy array (RGB)
        Returns:
            boxes (list): List of [x1, y1, x2, y2]
            labels (list): List of class indices
            scores (list): List of confidence scores
            severities (list): List of severity levels ('weak', 'moderate', 'heavy', or None)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Keep original image for cropping
        original_image = image.copy()
        
        # Transform image for detection
        img_tensor = F.to_tensor(image).to(self.device)
        
        # Use inference mode for better performance (PyTorch 1.9+)
        with torch.inference_mode():
            predictions = self.model([img_tensor])
            
        pred = predictions[0]
        
        # Filter by confidence (vectorized operation)
        mask = pred['scores'] > self.config.conf_threshold
        
        # Move to CPU and convert to numpy
        boxes = pred['boxes'][mask].cpu().numpy()
        labels_model = pred['labels'][mask].cpu().numpy()
        scores = pred['scores'][mask].cpu().numpy()
        
        # Remap labels from model output indices (1,2,3...) to global class IDs (0,1,3)
        # The model was trained with: global_to_model_idx = {0:1, 1:2, 3:3}
        # So model outputs 1,2,3 which we need to map back to 0,1,3
        labels_global = []
        for model_idx in labels_model:
            model_idx = int(model_idx)
            if model_idx in self.model_idx_to_global:
                labels_global.append(self.model_idx_to_global[model_idx])
            else:
                # Fallback: keep original if mapping not found
                labels_global.append(model_idx)
        
        # Classify severity for each detected box
        severities = []
        for box, class_id in zip(boxes, labels_global):
            try:
                # Crop the detected region
                x1, y1, x2, y2 = map(int, box)
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_image.width, x2)
                y2 = min(original_image.height, y2)
                
                # Skip if box is invalid
                if x2 <= x1 or y2 <= y1:
                    severities.append(None)
                    continue
                
                # Crop the image
                cropped = original_image.crop((x1, y1, x2, y2))
                
                # Predict severity
                severity = self.severity_classifier.predict_severity(cropped, class_id)
                severities.append(severity)
                
            except Exception as e:
                print(f"[ModelWrapper] Error classifying severity for box: {e}")
                severities.append(None)
        
        return boxes.tolist(), labels_global, scores.tolist(), severities
