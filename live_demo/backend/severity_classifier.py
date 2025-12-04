"""
Severity Classification Module
Integrates SimCLR + K-means clustering to classify damage severity
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import joblib
from PIL import Image
import numpy as np

# --------------------
# Model Definitions (from clustering.py)
# --------------------
class Identity(nn.Module):
    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True, use_bn=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias and not use_bn)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            LinearLayer(in_features, hidden_features, use_bias=True, use_bn=True),
            nn.ReLU(),
            LinearLayer(hidden_features, out_features, use_bias=False, use_bn=True)
        )

    def forward(self, x):
        return self.layers(x)

class PreModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 backbone
        self.pretrained = models.resnet18(pretrained=False)  # Don't download weights here
        # remove maxpool and adjust conv1
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pretrained.maxpool = Identity()
        # Remove final FC layer
        self.pretrained.fc = Identity()
        
        self.projector = ProjectionHead(
            in_features=512,
            hidden_features=512,
            out_features=128
        )

    def forward(self, x):
        h = self.pretrained(x)          # (B, 512)
        z = self.projector(h)           # (B, 128)
        return h, F.normalize(z, dim=1)


# --------------------
# Severity Classifier
# --------------------
class SeverityClassifier:
    """
    Loads SimCLR model + K-means for severity classification
    Supports multiple classes with different checkpoints
    """
    
    # Mapping from global class ID to checkpoint folder name
    CLASS_CHECKPOINT_MAP = {
        0: 'damageroad',  # Damaged Road
        1: 'pothole',     # Pothole
        3: 'roadsign'     # Broken Sign
    }
    
    # Severity level mapping (cluster ID -> severity label)
    CLUSTER_TO_SEVERITY = {
        0: "weak",
        1: "moderate",
        2: "heavy"
    }
    
    def __init__(self, checkpoint_base_dir='clustering/clustering_checkpoints', device=None):
        """
        Initialize severity classifier
        
        Args:
            checkpoint_base_dir: Base directory containing clustering checkpoints
            device: torch device (cuda or cpu)
        """
        self.checkpoint_base_dir = checkpoint_base_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache for loaded models and kmeans (class_id -> (model, kmeans))
        self.models = {}
        self.kmeans = {}
        
        # Image transform (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        print(f"[SeverityClassifier] Initialized on device: {self.device}")
    
    def _load_model_for_class(self, class_id):
        """Load SimCLR model and K-means for a specific class"""
        if class_id in self.models:
            return self.models[class_id], self.kmeans[class_id]
        
        if class_id not in self.CLASS_CHECKPOINT_MAP:
            print(f"[SeverityClassifier] No checkpoint for class {class_id}, skipping severity")
            return None, None
        
        checkpoint_folder = self.CLASS_CHECKPOINT_MAP[class_id]
        simclr_path = os.path.join(self.checkpoint_base_dir, checkpoint_folder, 
                                   f'simclr_best_adam_{checkpoint_folder}.pth')
        kmeans_path = os.path.join(self.checkpoint_base_dir, checkpoint_folder,
                                   f'kmeans_severity_{checkpoint_folder}.pkl')
        
        # Check if checkpoints exist
        if not os.path.exists(simclr_path) or not os.path.exists(kmeans_path):
            print(f"[SeverityClassifier] Missing checkpoints for class {class_id} ({checkpoint_folder})")
            print(f"  SimCLR: {simclr_path} (exists: {os.path.exists(simclr_path)})")
            print(f"  K-means: {kmeans_path} (exists: {os.path.exists(kmeans_path)})")
            return None, None
        
        try:
            # Load SimCLR model
            model = PreModel().to(self.device)
            model.load_state_dict(torch.load(simclr_path, map_location=self.device))
            model.eval()
            
            # Load K-means
            kmeans = joblib.load(kmeans_path)
            
            # Cache
            self.models[class_id] = model
            self.kmeans[class_id] = kmeans
            
            print(f"[SeverityClassifier] Loaded model for class {class_id} ({checkpoint_folder})")
            return model, kmeans
            
        except Exception as e:
            print(f"[SeverityClassifier] Error loading model for class {class_id}: {e}")
            return None, None
    
    def predict_severity(self, image_crop, class_id):
        """
        Predict severity for a cropped image of detected object
        
        Args:
            image_crop: PIL Image or numpy array (RGB)
            class_id: Global class ID (0=damaged road, 1=pothole, 3=broken sign)
        
        Returns:
            severity: str ('weak', 'moderate', 'heavy') or None if not available
        """
        # Load model for this class
        model, kmeans = self._load_model_for_class(class_id)
        
        if model is None or kmeans is None:
            return None
        
        try:
            # Convert to PIL if needed
            if isinstance(image_crop, np.ndarray):
                image_crop = Image.fromarray(image_crop)
            
            # Transform and add batch dimension
            tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                h, _ = model(tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                h_np = h.cpu().numpy()
            
            # Predict cluster
            cluster_id = kmeans.predict(h_np)[0]
            
            # Map to severity
            severity = self.CLUSTER_TO_SEVERITY.get(int(cluster_id), f"cluster_{cluster_id}")
            
            return severity
            
        except Exception as e:
            print(f"[SeverityClassifier] Error predicting severity for class {class_id}: {e}")
            return None
    
    def predict_batch(self, image_crops, class_ids):
        """
        Predict severity for multiple crops (more efficient)
        
        Args:
            image_crops: List of PIL Images or numpy arrays
            class_ids: List of class IDs corresponding to each crop
        
        Returns:
            List of severity labels (str or None)
        """
        severities = []
        
        for crop, class_id in zip(image_crops, class_ids):
            severity = self.predict_severity(crop, class_id)
            severities.append(severity)
        
        return severities

