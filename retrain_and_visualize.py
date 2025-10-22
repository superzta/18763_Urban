"""
Retrain and Visualize - Complete workflow for urban issue detection
Supports training on single or multiple urban issue classes (0-9)
"""

from rcnn import Config, train, test
import torch
import os
import shutil

def main():
    print("=" * 80)
    print("RUIDR - Urban Issue Detection Training")
    print("=" * 80)
    print("\nAvailable urban issue classes (0-9):")
    print("  0: Damaged Road issues")
    print("  1: Pothole Issues")
    print("  2: Illegal Parking Issues")
    print("  3: Broken Road Sign Issues (DEFAULT)")
    print("  4: Fallen trees")
    print("  5: Littering/Garbage on Public Places")
    print("  6: Vandalism Issues")
    print("  7: Dead Animal Pollution")
    print("  8: Damaged concrete structures")
    print("  9: Damaged Electric wires and poles")
    print()
    
    # Select classes to train on
    print("Enter class IDs to train on (comma-separated, e.g., '3' or '0,1,3'):")
    print("Press Enter for default (class 3 - Broken Road Signs)")
    user_input = input("Classes: ").strip()
    
    if user_input:
        try:
            urban_classes = [int(x.strip()) for x in user_input.split(',')]
        except:
            print("Invalid input, using default [3]")
            urban_classes = [3]
    else:
        urban_classes = [3]
    
    print(f"\nSelected classes: {urban_classes}")
    
    # Clean old results
    if os.path.exists('checkpoints'):
        response = input("\nDelete old checkpoints? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree('checkpoints')
            os.makedirs('checkpoints')
            print("✓ Deleted old checkpoints")
    
    # Create config with selected classes
    config = Config(urban_issue_classes=urban_classes)
    config.num_epochs = 1  # Quick training for demo
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

