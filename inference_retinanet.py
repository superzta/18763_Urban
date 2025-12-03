"""
RetinaNet Inference Script
==========================
Script to run inference using RetinaNet model.
"""

import os
from pathlib import Path
from datetime import datetime
from rcnn import Config
from retinanet import inference

def main():
    print("=" * 80)
    print("RetinaNet Inference Tool")
    print("=" * 80)
    
    # Configuration
    print("\nStep 1: Model Configuration")
    print("-" * 80)
    
    user_input = input("\nEnter class IDs your model was trained on (e.g., '3' or '0,1,3'): ").strip()
    if user_input:
        try:
            urban_classes = [int(x.strip()) for x in user_input.split(',')]
        except:
            print("Invalid input, using default [3]")
            urban_classes = [3]
    else:
        urban_classes = [3]
    
    config = Config(urban_issue_classes=urban_classes)
    
    conf_input = input(f"Confidence threshold (0-1, default {config.conf_threshold}): ").strip()
    if conf_input:
        try:
            config.conf_threshold = float(conf_input)
        except:
            pass
            
    checkpoint_input = input("Checkpoint path (default: checkpoints/retinanet_best_model.pth): ").strip()
    checkpoint_path = checkpoint_input if checkpoint_input else "checkpoints/retinanet_best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        return

    # Image Selection
    print("\nStep 2: Image Selection")
    print("-" * 80)
    image_path = input("\nEnter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Run Inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    save_path = os.path.join(config.output_dir, f'{image_name}_retinanet_{timestamp}.jpg')
    
    print(f"\nProcessing: {Path(image_path).name}")
    boxes, labels, scores = inference(config, checkpoint_path, image_path, save_path)
    
    print(f"\n✓ Detected {len(boxes)} objects:")
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        print(f"  [{i}] Confidence: {score:.3f}")
    
    print(f"\n✓ Output saved to: {save_path}")

if __name__ == '__main__':
    main()
