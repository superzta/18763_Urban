"""
Inference Example Script
=========================
Simple script to run inference on your own images or test images.

Usage:
    python inference_example.py
"""

import os
from pathlib import Path
from datetime import datetime
from rcnn import Config, inference


def inference_on_custom_image(config, checkpoint_path):
    """Run inference on user-provided image."""
    print("\n" + "=" * 80)
    print("Custom Image Inference")
    print("=" * 80)
    
    # Get image path from user
    image_path = input("\nEnter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    save_path = os.path.join(config.output_dir, f'{image_name}_inference_{timestamp}.jpg')
    
    # Run inference
    print(f"\nProcessing: {Path(image_path).name}")
    print(f"Confidence threshold: {config.conf_threshold}")
    
    boxes, labels, scores = inference(config, checkpoint_path, image_path, save_path)
    
    # Display results
    print(f"\n✓ Detected {len(boxes)} objects:")
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        print(f"  [{i}] Confidence: {score:.3f}")
    
    print(f"\n✓ Output saved to: {save_path}")
    print("  (Image includes bounding boxes and labels)")


def inference_on_test_images(config, checkpoint_path):
    """Run inference on random test images from trained classes."""
    print("\n" + "=" * 80)
    print("Test Images Inference")
    print("=" * 80)
    
    # Ask how many images per class
    num_input = input("\nNumber of random images per class to test (default 3): ").strip()
    num_per_class = int(num_input) if num_input else 3
    
    all_test_images = []
    
    # Collect test images from each trained class
    for class_id in config.urban_issue_classes:
        folder_name = config.URBAN_ISSUE_DATASETS[class_id][0]
        class_name = config.URBAN_ISSUE_DATASETS[class_id][1]
        test_dir = Path(f"data/{folder_name}/{folder_name}/test/images")
        
        if test_dir.exists():
            images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
            if images:
                # Select random images
                import random
                selected = random.sample(images, min(num_per_class, len(images)))
                all_test_images.extend([(img, class_id, class_name) for img in selected])
                print(f"  Class {class_id} ({class_name}): {len(selected)} images")
    
    if not all_test_images:
        print("No test images found for the selected classes.")
        return
    
    print(f"\nTotal: {len(all_test_images)} images")
    print(f"Confidence threshold: {config.conf_threshold}")
    
    input("\nPress Enter to start inference...")
    
    # Process each image
    total_detections = 0
    
    for i, (img_path, class_id, class_name) in enumerate(all_test_images, 1):
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(config.output_dir, f'test_class{class_id}_{i}_{timestamp}.jpg')
        
        # Run inference
        print(f"\n[{i}/{len(all_test_images)}] {img_path.name}")
        print(f"  Ground truth class: {class_name}")
        
        boxes, labels, scores = inference(config, checkpoint_path, str(img_path), save_path)
        
        print(f"  ✓ Detected {len(boxes)} objects")
        for j, (label, score) in enumerate(zip(labels, scores), 1):
            print(f"    [{j}] Confidence: {score:.3f}")
        print(f"  ✓ Saved to: {save_path}")
        
        total_detections += len(boxes)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"✓ Inference completed!")
    print(f"  Total images: {len(all_test_images)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average: {total_detections / len(all_test_images):.2f} detections/image")
    print(f"  Outputs: {config.output_dir}/")


def main():
    """Main inference script."""
    print("=" * 80)
    print("RUIDR - Inference Tool")
    print("=" * 80)
    
    # Step 1: Configure model
    print("\nStep 1: Model Configuration")
    print("-" * 80)
    
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
    
    user_input = input("\nEnter class IDs your model was trained on (e.g., '3' or '0,1,3'): ").strip()
    
    if user_input:
        try:
            urban_classes = [int(x.strip()) for x in user_input.split(',')]
        except:
            print("Invalid input, using default [3]")
            urban_classes = [3]
    else:
        urban_classes = [3]
    
    print(f"Selected classes: {urban_classes}")
    
    # Create config
    config = Config(urban_issue_classes=urban_classes)
    
    # Confidence threshold
    conf_input = input(f"Confidence threshold (0-1, default {config.conf_threshold}): ").strip()
    if conf_input:
        try:
            config.conf_threshold = float(conf_input)
        except:
            print(f"Invalid input, using default {config.conf_threshold}")
    
    # Checkpoint path
    checkpoint_input = input("Checkpoint path (default: checkpoints/best_model.pth): ").strip()
    checkpoint_path = checkpoint_input if checkpoint_input else "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify a valid checkpoint path.")
        return
    
    print(f"\n✓ Configuration complete")
    print(f"  Classes: {urban_classes}")
    print(f"  Confidence: {config.conf_threshold}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Step 2: Select inference mode
    print("\n\nStep 2: Select Inference Mode")
    print("-" * 80)
    print("  1. Inference on YOUR OWN image (MAIN FEATURE)")
    print("  2. Test on random images from dataset")
    
    choice = input("\nChoose mode (1/2, default 1): ").strip() or "1"
    
    if choice == "2":
        inference_on_test_images(config, checkpoint_path)
    else:
        inference_on_custom_image(config, checkpoint_path)
    
    print("\n" + "=" * 80)
    print("Done! Check the output images in: " + config.output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
