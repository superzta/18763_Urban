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

class ModelWrapper:
    def __init__(self, model_type='rcnn', checkpoint_path=None, classes=[3], conf_threshold=0.5, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config(urban_issue_classes=classes)
        self.config.conf_threshold = conf_threshold
        self.model_type = model_type
        
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
        
    def predict(self, image):
        """
        Run inference on a single image.
        Args:
            image: PIL Image or numpy array (RGB)
        Returns:
            boxes (list): List of [x1, y1, x2, y2]
            labels (list): List of class indices
            scores (list): List of confidence scores
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Transform image
        img_tensor = F.to_tensor(image).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([img_tensor])
            
        pred = predictions[0]
        
        # Filter by confidence
        mask = pred['scores'] > self.config.conf_threshold
        boxes = pred['boxes'][mask].cpu().numpy().tolist()
        labels = pred['labels'][mask].cpu().numpy().tolist()
        scores = pred['scores'][mask].cpu().numpy().tolist()
        
        return boxes, labels, scores
