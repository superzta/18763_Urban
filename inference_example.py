"""
Inference Example Script
=========================
This script demonstrates how to run inference programmatically
and process results for downstream tasks.
"""

import os
from pathlib import Path
import torch
from PIL import Image
from rcnn import Config, get_model, inference
from torchvision.transforms import functional as F
import json


def batch_inference_with_results(config, checkpoint_path, image_dir, output_json=None):
    """
    Run inference on a directory of images and save structured results.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        image_dir: Directory containing images
        output_json: Optional path to save results as JSON
    """
    print("=" * 80)
    print("Batch Inference with Structured Results")
    print("=" * 80)
    
    # Load class names
    import yaml
    with open(config.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = ['background'] + data_config['names']
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Get all images
    image_dir = Path(image_dir)
    image_files = sorted(list(image_dir.glob('*.jpg')) + 
                        list(image_dir.glob('*.png')) + 
                        list(image_dir.glob('*.jpeg')))
    
    print(f"Found {len(image_files)} images\n")
    
    all_results = []
    
    # Process each image
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0).to(config.device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        # Filter by confidence
        mask = predictions['scores'] > config.conf_threshold
        boxes = predictions['boxes'][mask].cpu().numpy()
        labels = predictions['labels'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()
        
        # Structure results
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detection = {
                'class_id': int(label),
                'class_name': class_names[label],
                'confidence': float(score),
                'bbox': {
                    'x_min': float(box[0]),
                    'y_min': float(box[1]),
                    'x_max': float(box[2]),
                    'y_max': float(box[3])
                }
            }
            detections.append(detection)
        
        image_result = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'num_detections': len(detections),
            'detections': detections
        }
        
        all_results.append(image_result)
        
        print(f"  Detected {len(detections)} objects")
        for det in detections:
            print(f"    - {det['class_name']}: {det['confidence']:.3f}")
        print()
    
    # Save results
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_json}")
    
    # Summary statistics
    total_detections = sum(r['num_detections'] for r in all_results)
    class_counts = {}
    for result in all_results:
        for det in result['detections']:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(all_results):.2f}")
    print("\nDetections by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:20s}: {count}")
    print("=" * 80)
    
    return all_results


def single_image_inference_detailed(config, checkpoint_path, image_path):
    """
    Run inference on a single image with detailed output.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        image_path: Path to image
    """
    print("=" * 80)
    print("Single Image Inference")
    print("=" * 80)
    
    # Load class names
    import yaml
    with open(config.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = ['background'] + data_config['names']
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    print(f"  Image size: {image_width} x {image_height}")
    
    image_tensor = F.to_tensor(image).unsqueeze(0).to(config.device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # All predictions (before filtering)
    print(f"\nTotal predictions: {len(predictions['boxes'])}")
    
    # Filter by confidence
    mask = predictions['scores'] > config.conf_threshold
    boxes = predictions['boxes'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    
    print(f"Predictions above {config.conf_threshold} confidence: {len(boxes)}")
    print("\nDetailed results:")
    print("-" * 80)
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = class_names[label]
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        
        print(f"\nDetection {i+1}:")
        print(f"  Class: {class_name} (ID: {label})")
        print(f"  Confidence: {score:.4f} ({score*100:.2f}%)")
        print(f"  Bounding Box:")
        print(f"    Top-left: ({box[0]:.1f}, {box[1]:.1f})")
        print(f"    Bottom-right: ({box[2]:.1f}, {box[3]:.1f})")
        print(f"    Width: {box_width:.1f} px")
        print(f"    Height: {box_height:.1f} px")
        print(f"    Area: {box_area:.1f} px²")
        print(f"    % of image: {(box_area / (image_width * image_height)) * 100:.2f}%")
    
    print("\n" + "=" * 80)
    
    return boxes, labels, scores


def main():
    """Example usage."""
    # Create configuration
    config = Config()
    
    # Set checkpoint path
    checkpoint_path = "checkpoints/best_model.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify a valid checkpoint path.")
        return
    
    # ========================================================================
    # EXAMPLE 1: Single image inference with detailed output
    # ========================================================================
    print("\n\nEXAMPLE 1: Single Image Inference\n")
    
    test_image = "data/DamagedRoadSigns/DamagedRoadSigns/test/images"
    if os.path.exists(test_image):
        # Get first image in test directory
        image_files = list(Path(test_image).glob('*.jpg'))
        if image_files:
            single_image_inference_detailed(config, checkpoint_path, str(image_files[0]))
    
    # ========================================================================
    # EXAMPLE 2: Batch inference with structured results
    # ========================================================================
    print("\n\nEXAMPLE 2: Batch Inference\n")
    
    test_dir = "data/DamagedRoadSigns/DamagedRoadSigns/test/images"
    if os.path.exists(test_dir):
        results = batch_inference_with_results(
            config, 
            checkpoint_path, 
            test_dir,
            output_json="results/inference_results.json"
        )
    
    # ========================================================================
    # EXAMPLE 3: Filter results by confidence threshold
    # ========================================================================
    print("\n\nEXAMPLE 3: Filtering by Confidence\n")
    
    # Try different confidence thresholds
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        config.conf_threshold = threshold
        
        if os.path.exists(test_dir):
            image_files = list(Path(test_dir).glob('*.jpg'))[:5]  # First 5 images
            
            total_detections = 0
            for img_path in image_files:
                image = Image.open(img_path).convert('RGB')
                image_tensor = F.to_tensor(image).unsqueeze(0).to(config.device)
                
                model = get_model(config.num_classes, pretrained=False)
                checkpoint = torch.load(checkpoint_path, map_location=config.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(config.device)
                model.eval()
                
                with torch.no_grad():
                    predictions = model(image_tensor)[0]
                
                mask = predictions['scores'] > threshold
                total_detections += mask.sum().item()
            
            avg_detections = total_detections / len(image_files)
            print(f"Threshold {threshold:.1f}: {avg_detections:.2f} detections/image")


if __name__ == '__main__':
    main()

