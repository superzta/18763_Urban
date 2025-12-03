"""
RetinaNet Testing Examples
==========================
Shows how to test RetinaNet on different combinations of urban issue classes.
"""

from rcnn import Config
from retinanet import test

def example_1_single_class(checkpoint_path='checkpoints/retinanet_best_model.pth'):
    """Example 1: Test on single class (Broken Road Signs)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Testing RetinaNet on Single Class (Broken Road Signs)")
    print("="*80)
    
    config = Config(urban_issue_classes=[3])  # Class 3: Broken Road Sign Issues
    config.conf_threshold = 0.5
    
    results = test(config, checkpoint_path)
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

def example_2_road_issues(checkpoint_path='checkpoints/retinanet_best_model.pth'):
    """Example 2: Test on all road-related issues"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Testing RetinaNet on Road-Related Issues")
    print("="*80)
    
    # Road damage (0), Potholes (1), Broken Road Signs (3)
    config = Config(urban_issue_classes=[0, 1, 3])
    
    # Allow user to override threshold
    user_thresh = input("Enter confidence threshold (default: 0.5): ").strip()
    if user_thresh:
        config.conf_threshold = float(user_thresh)
    else:
        config.conf_threshold = 0.5
    
    results = test(config, checkpoint_path)
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print("\nVisualizations have been saved to the 'results' directory.")
    print("Check for files named 'eval_comparison_{ClassName}_{i}.png'")


def example_5_all_classes(checkpoint_path='checkpoints/retinanet_best_model.pth'):
    """Example 5: Test on ALL urban issue classes"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Testing RetinaNet on ALL Classes (Comprehensive Model)")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    config.conf_threshold = 0.5
    
    results = test(config, checkpoint_path)
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


if __name__ == '__main__':
    import os
    import sys
    
    print("="*80)
    print("RetinaNet Testing Examples")
    print("="*80)
    
    checkpoint_input = input("\nEnter checkpoint path (default: checkpoints/retinanet_best_model.pth): ").strip()
    checkpoint_path = checkpoint_input if checkpoint_input else 'checkpoints/retinanet_best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠ Warning: {checkpoint_path} not found!")
        print("Please train a model first.")
        sys.exit(0)
    
    print(f"\nUsing checkpoint: {checkpoint_path}")
    print("\nAvailable examples:")
    print("  1. Single class (Broken Road Signs)")
    print("  2. Road-related issues (Damaged roads, Potholes, Signs)")
    print("  5. ALL classes (Comprehensive model)")
    print()
    
    choice = input("Select example (1, 2, 5) or Enter for default (1): ").strip()
    
    if choice == '1' or choice == '':
        example_1_single_class(checkpoint_path)
    elif choice == '2':
        example_2_road_issues(checkpoint_path)
    elif choice == '5':
        example_5_all_classes(checkpoint_path)
    else:
        print("Invalid choice, running default (Example 1)")
        example_1_single_class(checkpoint_path)
