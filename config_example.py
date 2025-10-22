"""
Example Configuration File
===========================
Copy and modify this file to customize your training settings.

You can edit the Config class in rcnn.py directly, or create a custom
config and modify the Config class initialization.
"""

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

DATA_ROOT = "data/DamagedRoadSigns/DamagedRoadSigns"
DATA_YAML = "data/DamagedRoadSigns/DamagedRoadSigns/data.yaml"

# For different datasets, update these paths:
# DATA_ROOT = "data/Damaged concrete structures/Damaged concrete structures"
# DATA_ROOT = "data/DamagedElectricalPoles/DamagedElectricalPoles"


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Number of classes (background + your object classes)
# For DamagedRoadSigns: 2 classes (Damage, Healthy) + 1 background = 3
NUM_CLASSES = 3

# Backbone architecture
BACKBONE = "resnet50"  # Options: resnet50, resnet101

# Use pretrained weights (highly recommended)
PRETRAINED = True


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Batch size - adjust based on your GPU memory
# GPU Memory:  Recommended batch size:
# 4GB          1-2
# 8GB          4-8
# 12GB+        8-16
BATCH_SIZE = 4

# Number of training epochs
# Quick test: 5-10
# Standard: 50
# High quality: 100+
NUM_EPOCHS = 50

# Learning rate
# Start with 0.005, adjust if needed:
# - If loss doesn't decrease: try 0.001 or 0.002
# - If loss oscillates: try 0.01 or 0.02
LEARNING_RATE = 0.005

# SGD optimizer parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Learning rate scheduler
# Multiply LR by GAMMA every STEP_SIZE epochs
LR_SCHEDULER_STEP_SIZE = 10  # Decay every N epochs
LR_SCHEDULER_GAMMA = 0.1     # Multiply LR by this factor


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

# Probability of horizontal flip augmentation
HORIZONTAL_FLIP_PROB = 0.5

# Image size for training
MIN_SIZE = 640
MAX_SIZE = 640


# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================

# Number of data loading workers
# Increase for faster data loading (if you have enough CPU cores)
NUM_WORKERS = 4

# Print frequency during training
PRINT_FREQ = 50

# Save checkpoint every N epochs
SAVE_FREQ = 5


# ==============================================================================
# INFERENCE SETTINGS
# ==============================================================================

# Confidence threshold for detections
# Higher = fewer but more confident detections
# Lower = more detections but may include false positives
CONF_THRESHOLD = 0.5

# Non-Maximum Suppression (NMS) IoU threshold
# Lower = more aggressive NMS (fewer overlapping boxes)
# Higher = less aggressive NMS (may keep overlapping boxes)
NMS_THRESHOLD = 0.3


# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
RESULTS_DIR = "results"


# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

# Fast experimentation (5-10 minutes on GPU)
FAST_CONFIG = {
    'num_epochs': 5,
    'batch_size': 2,
    'learning_rate': 0.005,
    'save_freq': 2
}

# Standard training (1-2 hours on GPU)
STANDARD_CONFIG = {
    'num_epochs': 50,
    'batch_size': 4,
    'learning_rate': 0.005,
    'save_freq': 5
}

# High-quality training (4-8 hours on GPU)
HIGH_QUALITY_CONFIG = {
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 0.003,
    'save_freq': 10
}

# Low-memory configuration (for 4GB GPU)
LOW_MEMORY_CONFIG = {
    'num_epochs': 50,
    'batch_size': 1,
    'learning_rate': 0.005,
    'num_workers': 2
}


# ==============================================================================
# TIPS FOR HYPERPARAMETER TUNING
# ==============================================================================

"""
1. LEARNING RATE:
   - Too high: Loss will oscillate or diverge
   - Too low: Training will be very slow
   - Start with 0.005 and adjust based on training curves

2. BATCH SIZE:
   - Larger batch: More stable gradients, faster training
   - Smaller batch: More noisy gradients, may generalize better
   - Limited by GPU memory

3. NUMBER OF EPOCHS:
   - Monitor validation loss to avoid overfitting
   - Early stopping: stop when validation loss stops improving

4. LEARNING RATE SCHEDULE:
   - StepLR: Decay by gamma every step_size epochs
   - Helps fine-tune the model in later epochs

5. DATA AUGMENTATION:
   - Horizontal flip: Good for most object detection tasks
   - Can add more augmentations in the dataset class

6. CONFIDENCE THRESHOLD:
   - For inference/evaluation only
   - Higher threshold = more precise but may miss objects
   - Lower threshold = more recall but more false positives
   - Adjust based on your use case (precision vs recall trade-off)
"""

