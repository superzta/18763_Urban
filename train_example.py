"""
Training Example Script
========================
This is a simple example showing how to train the model programmatically
instead of using command-line arguments.

You can copy and modify this script for custom training workflows.
"""

import torch
from rcnn import Config, train


def main():
    """Example training with custom configuration."""
    
    # Create configuration
    config = Config()
    
    # ========================================================================
    # CUSTOMIZE YOUR TRAINING HERE
    # ========================================================================
    
    # Dataset paths
    config.data_root = "data/DamagedRoadSigns/DamagedRoadSigns"
    config.data_yaml = "data/DamagedRoadSigns/DamagedRoadSigns/data.yaml"
    
    # Model settings
    config.num_classes = 3  # background + 2 classes
    config.pretrained = True  # Use ImageNet pretrained weights
    
    # Training hyperparameters
    config.num_epochs = 50
    config.batch_size = 4
    config.learning_rate = 0.005
    config.momentum = 0.9
    config.weight_decay = 0.0005
    
    # Learning rate scheduler
    config.lr_scheduler_step_size = 10  # Decay LR every 10 epochs
    config.lr_scheduler_gamma = 0.1     # Multiply LR by 0.1
    
    # Data augmentation
    config.horizontal_flip_prob = 0.5
    config.min_size = 640
    config.max_size = 640
    
    # Training settings
    config.num_workers = 4
    config.print_freq = 50
    config.save_freq = 5  # Save checkpoint every 5 epochs
    
    # Output directories
    config.checkpoint_dir = "checkpoints"
    config.output_dir = "outputs"
    config.results_dir = "results"
    
    # ========================================================================
    # PRINT CONFIGURATION
    # ========================================================================
    
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Dataset: {config.data_root}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Pretrained: {config.pretrained}")
    print("=" * 80)
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA is not available, training will use CPU (slower)")
    print()
    
    # ========================================================================
    # START TRAINING
    # ========================================================================
    
    model = train(config)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Best model saved to: {config.checkpoint_dir}/best_model.pth")
    print(f"Training curves saved to: {config.results_dir}/training_curves.png")
    print(f"Training history saved to: {config.results_dir}/training_history.json")
    print()
    print("Next steps:")
    print("1. Check training curves to evaluate model performance")
    print("2. Test the model: python rcnn.py --mode test --checkpoint checkpoints/best_model.pth")
    print("3. Run inference: python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image <path>")
    print("=" * 80)


if __name__ == '__main__':
    main()

