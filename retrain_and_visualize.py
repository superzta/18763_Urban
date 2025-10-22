"""
Retrain and Visualize - Complete workflow with proper class ID mapping
"""

from rcnn import Config, train, test
import torch
import os
import shutil

def main():
    print("=" * 80)
    print("RUIDR - Retrain with Fixed Class ID Mapping")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Train model with corrected class ID mapping (3 → 0)")
    print("  2. Test on test set")
    print("  3. Generate visualizations comparing predictions vs ground truth")
    print()
    
    # Clean old results
    if os.path.exists('checkpoints'):
        response = input("Delete old checkpoints? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree('checkpoints')
            os.makedirs('checkpoints')
            print("✓ Deleted old checkpoints")
    
    # Create config
    config = Config()
    config.num_epochs = 2  # Quick retrain for demo
    config.batch_size = 4
    config.learning_rate = 0.005
    config.conf_threshold = 0.3  # Lower threshold to see more detections
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Confidence threshold: {config.conf_threshold}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (slower)")
    print()
    
    # Train
    print("=" * 80)
    print("STEP 1: Training")
    print("=" * 80)
    model = train(config)
    
    # Test
    print("\n" + "=" * 80)
    print("STEP 2: Testing with Visualizations")
    print("=" * 80)
    results = test(config, 'checkpoints/best_model.pth')
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  Checkpoints:")
    print("    - checkpoints/best_model.pth")
    print("  Results:")
    print("    - results/training_curves.png (loss curves)")
    print("    - results/training_history.json (detailed logs)")
    print("    - results/evaluation_results.json (mAP, precision, recall)")
    print("    - results/eval_comparison_*.png (GT vs Predictions)")
    print()
    print("Next: View the eval_comparison images to see how the model performs!")
    print("=" * 80)


if __name__ == '__main__':
    main()

