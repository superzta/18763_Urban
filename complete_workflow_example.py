"""
Complete Workflow Example
==========================
This script demonstrates the complete workflow from training to inference.
Run this to see the entire pipeline in action.

Usage:
    python complete_workflow_example.py
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_prerequisites():
    """Check if all prerequisites are met."""
    print_section("STEP 1: Checking Prerequisites")
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check if packages are installed
    packages_ok = True
    required_packages = ['torch', 'torchvision', 'PIL', 'yaml', 'matplotlib']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:15s} - Installed")
        except ImportError:
            print(f"✗ {package:15s} - Missing")
            packages_ok = False
    
    if not packages_ok:
        print("\n Some packages are missing!")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check dataset
    dataset_path = "data/DamagedRoadSigns/DamagedRoadSigns"
    if not os.path.exists(dataset_path):
        print(f"\n Dataset not found at: {dataset_path}")
        print("Please download and extract the dataset.")
        return False
    
    print(f"\n✓ Dataset found at: {dataset_path}")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available - Will use CPU (slower)")
    
    print("\n All prerequisites met!")
    return True


def demo_training():
    """Demonstrate training with minimal epochs."""
    print_section("STEP 2: Training Demo (Quick Training)")
    
    print("Available urban issue classes (0-9):")
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
    
    # Ask for number of epochs
    epochs_input = input("Number of epochs (default 5 for demo): ").strip()
    num_epochs = int(epochs_input) if epochs_input else 5
    
    print(f"\nThis will train a model for {num_epochs} epochs.")
    print("For production use, train for 50-100 epochs.\n")
    
    input("Press Enter to start training, or Ctrl+C to skip...")
    
    # Import here to avoid issues if packages not installed
    from rcnn import Config, train
    
    # Create config for demo
    config = Config(urban_issue_classes=urban_classes)
    config.num_epochs = num_epochs
    config.batch_size = 2  # Small batch for compatibility
    config.save_freq = 2   # Save more frequently
    config.print_freq = 10  # Print more frequently
    
    print("\nStarting training...")
    print(f"Classes: {config.urban_issue_classes}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print()
    
    try:
        model = train(config)
        print("\n Training completed successfully!")
        return config, True
    except Exception as e:
        print(f"\n Training failed: {str(e)}")
        return None, False


def demo_evaluation(config):
    """Demonstrate model evaluation."""
    print_section("STEP 3: Model Evaluation")
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        return False
    
    print(f"Evaluating model: {checkpoint_path}")
    print(f"Testing on classes: {config.urban_issue_classes}\n")
    
    input("Press Enter to start evaluation, or Ctrl+C to skip...")
    
    from rcnn import test
    
    try:
        results = test(config, checkpoint_path)
        print("\n Evaluation completed successfully!")
        print(f"   mAP@0.5: {results['mAP@0.5']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        return True
    except Exception as e:
        print(f"\n Evaluation failed: {str(e)}")
        return False


def demo_inference(config):
    """Demonstrate inference on test images with visualization."""
    print_section("STEP 4: Inference Demo")
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        return False
    
    # Ask user for image path or use test images
    print("Options:")
    print("  1. Use test images from dataset")
    print("  2. Specify custom image path")
    print("  3. Specify custom directory path")
    choice = input("Choose option (1/2/3, default 1): ").strip() or "1"
    
    if choice == "2":
        image_path = input("Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f" Image not found: {image_path}")
            return False
        image_files = [Path(image_path)]
    elif choice == "3":
        dir_path = input("Enter directory path: ").strip()
        if not os.path.exists(dir_path):
            print(f" Directory not found: {dir_path}")
            return False
        image_files = list(Path(dir_path).glob('*.jpg')) + list(Path(dir_path).glob('*.png'))
        if not image_files:
            print(" No images found in directory")
            return False
        # Limit to first 5 images for demo
        image_files = image_files[:5]
    else:
        # Use test images from first selected class
        folder_name = config.URBAN_ISSUE_DATASETS[config.urban_issue_classes[0]][0]
        test_image_dir = f"data/{folder_name}/{folder_name}/test/images"
        
        if not os.path.exists(test_image_dir):
            print(f" Test images not found: {test_image_dir}")
            return False
        
        # Get first 5 test images
        image_files = list(Path(test_image_dir).glob('*.jpg'))[:5]
        if not image_files:
            image_files = list(Path(test_image_dir).glob('*.png'))[:5]
        
        if not image_files:
            print(" No test images found.")
            return False
    
    # Ask for confidence threshold
    conf_input = input(f"Confidence threshold (0-1, default {config.conf_threshold}): ").strip()
    if conf_input:
        try:
            config.conf_threshold = float(conf_input)
        except:
            print(f"Invalid input, using default {config.conf_threshold}")
    
    print(f"\nRunning inference on {len(image_files)} image(s)")
    print(f"Confidence threshold: {config.conf_threshold}\n")
    
    input("Press Enter to run inference, or Ctrl+C to skip...")
    
    from rcnn import inference
    from datetime import datetime
    
    total_detections = 0
    
    try:
        for i, img_path in enumerate(image_files, 1):
            print(f"\nProcessing image {i}/{len(image_files)}: {img_path.name}")
            
            # Create output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(config.output_dir, f'demo_inference_{i}_{timestamp}.jpg')
            
            boxes, labels, scores = inference(config, checkpoint_path, str(img_path), save_path)
            
            print(f"✓ Detected {len(boxes)} objects")
            for j, (box, label, score) in enumerate(zip(boxes, labels, scores), 1):
                print(f"  [{j}] Confidence: {score:.3f}")
            print(f"✓ Saved to: {save_path}")
            
            total_detections += len(boxes)
        
        print(f"\n Inference completed!")
        print(f"   Processed: {len(image_files)} images")
        print(f"   Total detections: {total_detections}")
        print(f"   Average: {total_detections / len(image_files):.2f} detections/image")
        print(f"   Outputs saved to: {config.output_dir}/")
        
        return True
    except Exception as e:
        print(f"\n Inference failed: {str(e)}")
        return False


def show_results():
    """Show summary of results."""
    print_section("STEP 5: Results Summary")
    
    print("Generated files:\n")
    
    # Check checkpoints
    if os.path.exists("checkpoints"):
        checkpoints = list(Path("checkpoints").glob("*.pth"))
        print(f" Checkpoints ({len(checkpoints)} files):")
        for cp in sorted(checkpoints)[:5]:  # Show first 5
            size_mb = os.path.getsize(cp) / (1024 * 1024)
            print(f"   - {cp.name} ({size_mb:.1f} MB)")
        if len(checkpoints) > 5:
            print(f"   ... and {len(checkpoints) - 5} more")
        print()
    
    # Check results
    if os.path.exists("results"):
        results = list(Path("results").glob("*"))
        print(f" Results ({len(results)} files):")
        for result in sorted(results):
            print(f"   - {result.name}")
        print()
    
    # Check outputs
    if os.path.exists("outputs"):
        outputs = list(Path("outputs").glob("*.jpg"))
        print(f"  Inference Outputs ({len(outputs)} files):")
        for output in sorted(outputs)[:5]:  # Show first 5
            print(f"   - {output.name}")
        if len(outputs) > 5:
            print(f"   ... and {len(outputs) - 5} more")
        print()
    
    print("Next steps:")
    print("1. Review training curves: results/training_curves.png")
    print("2. Check evaluation results: results/evaluation_results.json")
    print("3. View inference outputs: outputs/*.jpg")
    print("4. Train longer for better results (50-100 epochs)")


def show_usage_examples():
    """Show usage examples."""
    print_section("Usage Examples")
    
    print("Command-line usage:\n")
    
    print("1. Training:")
    print("   python rcnn.py --mode train --epochs 50 --batch-size 4\n")
    
    print("2. Testing:")
    print("   python rcnn.py --mode test --checkpoint checkpoints/best_model.pth\n")
    
    print("3. Single image inference:")
    print("   python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image image.jpg\n")
    
    print("4. Batch inference:")
    print("   python rcnn.py --mode inference --checkpoint checkpoints/best_model.pth --image-dir images/\n")
    
    print("For more examples, see:")
    print("  - README.md")
    print("  - QUICKSTART.md")
    print("  - train_example.py")
    print("  - inference_example.py")


def main():
    """Main workflow demonstration."""
    print("\n" + "=" * 80)
    print("  RUIDR Faster R-CNN - Complete Workflow Demo")
    print("=" * 80)
    print("\nThis script will guide you through the complete workflow:")
    print("  1. Check prerequisites")
    print("  2. Train a demo model (5 epochs)")
    print("  3. Evaluate the model")
    print("  4. Run inference on a test image")
    print("  5. Show results")
    print("\nNote: This is a quick demo. For real use, train for 50-100 epochs.")
    
    try:
        # Step 1: Prerequisites
        if not check_prerequisites():
            print("\n Prerequisites check failed. Please fix issues and try again.")
            return
        
        # Ask if user wants to run full demo
        print("\n" + "-" * 80)
        response = input("\nRun full demo? This will train a model. (y/n): ").lower()
        
        if response != 'y':
            print("\nSkipping demo. Here are usage examples instead:")
            show_usage_examples()
            return
        
        # Step 2: Training
        config, success = demo_training()
        if not success:
            print("\n⚠ Training failed, skipping remaining steps.")
            show_usage_examples()
            return
        
        # Step 3: Evaluation
        print("\n" + "-" * 80)
        response = input("\nRun evaluation? (y/n): ").lower()
        if response == 'y':
            demo_evaluation(config)
        
        # Step 4: Inference
        print("\n" + "-" * 80)
        response = input("\nRun inference demo? (y/n): ").lower()
        if response == 'y':
            demo_inference(config)
        
        # Step 5: Show results
        show_results()
        
        print_section("Demo Completed!")
        print(" Workflow demo completed successfully!")
        print("\nYou can now:")
        print("  - Train with more epochs for better results")
        print("  - Experiment with different hyperparameters")
        print("  - Run inference on your own images")
        print("\nSee README.md for detailed documentation.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user.")
        print("\nYou can run individual steps using rcnn.py:")
        show_usage_examples()
    except Exception as e:
        print(f"\n\n Demo failed with error: {str(e)}")
        print("\nPlease check the error message and try again.")


if __name__ == '__main__':
    main()

