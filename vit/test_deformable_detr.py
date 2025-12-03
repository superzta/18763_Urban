
"""
Example Testing Scripts for Deformable DETR on Urban Issue Scenarios
====================================================================
Shows how to test Deformable DETR on different combinations of urban issue classes.
"""

from rcnn import Config
from deformable_detr import test

def example_1_single_class():
    """Example 1: Test on single class (Broken Road Signs)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Testing Deformable DETR on Single Class (Broken Road Signs)")
    print("="*80)
    
    config = Config(urban_issue_classes=[3])  # Class 3: Broken Road Sign Issues
    config.conf_threshold = 0.2  # Adjust as needed
    
    # Check if checkpoint exists
    import os
    checkpoint_path = 'checkpoints/deformable_detr_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Warning: {checkpoint_path} not found. Using best_model.pth if available or skipping.")
        checkpoint_path = 'checkpoints/best_model.pth' # Fallback
        if not os.path.exists(checkpoint_path):
             print("No checkpoint found. Please train first.")
             return

    results = test(config, checkpoint_path)
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


# def example_2_road_issues():
#     """Example 2: Test on all road-related issues"""
#     print("\n" + "="*80)
#     print("EXAMPLE 2: Testing Deformable DETR on Road-Related Issues")
#     print("="*80)
    
#     config = Config(urban_issue_classes=[0, 1, 3])  # Road damage, potholes, signs
#     config.conf_threshold = 0.5
    
#     results = test(config, 'checkpoints/deformable_detr_30_best.pth')
    
#     print("\n✓ Testing complete!")
#     print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
#     print(f"  Precision: {results['precision']:.4f}")
#     print(f"  Recall: {results['recall']:.4f}")


def example_2_road_issues():
    """Example 2: Test on all road-related issues"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Testing Deformable DETR on Road-Related Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 3])  # Road damage, potholes, signs
    config.conf_threshold = 0.5

    # IMPORTANT: Deformable DETR is heavy → use smaller batch size
    config.batch_size = 1      # or 2 if your GPU can handle it
    config.num_workers = 0     # optional; to avoid worker RAM issues on Windows

    results = test(config, 'checkpoints/deformable_detr_30_best.pth')
    
    print("\nTesting complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")



def example_3_infrastructure():
    """Example 3: Test on infrastructure issues"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Testing Deformable DETR on Infrastructure Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[3, 8, 9])  # Signs, Concrete, Electrical
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/deformable_detr_epoch_15.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_4_environmental():
    """Example 4: Test on environmental issues"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Testing Deformable DETR on Environmental Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[4, 5, 7])  # Trees, garbage, dead animals
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/deformable_detr_epoch_15.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_5_all_classes():
    """Example 5: Test on ALL urban issue classes"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Testing Deformable DETR on ALL Classes")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/deformable_detr_epoch_20.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def custom_testing(class_list, checkpoint_path, conf_threshold=0.5):
    """Custom testing with specified classes"""
    print("\n" + "="*80)
    print(f"CUSTOM: Testing Deformable DETR on Classes {class_list}")
    print("="*80)
    
    config = Config(urban_issue_classes=class_list)
    config.conf_threshold = conf_threshold
    
    results = test(config, checkpoint_path)
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    return results


if __name__ == '__main__':
    import os
    import sys
    
    print("="*80)
    print("Deformable DETR Testing Examples")
    print("="*80)
    
    # Check if checkpoint exists
    # We'll ask user for checkpoint path or assume one based on example
    
    print("\nAvailable examples:")
    print("  1. Single class (Broken Road Signs)")
    print("  2. Road-related issues (Damaged roads, Potholes, Signs)")
    print("  3. Infrastructure (Concrete structures, Electrical poles, Signs)")
    print("  4. Environmental (Trees, Garbage, Dead animals)")
    print("  5. ALL classes (Comprehensive model)")
    print("  6. Custom (specify your own classes)")
    print()
    
    choice = input("Select example (1-6) or Enter for default (1): ").strip()
    
    if choice == '1' or choice == '':
        example_1_single_class()
    elif choice == '2':
        example_2_road_issues()
    elif choice == '3':
        example_3_infrastructure()
    elif choice == '4':
        example_4_environmental()
    elif choice == '5':
        example_5_all_classes()
    elif choice == '6':
        print("\nAvailable classes:")
        print("  0: Damaged Road issues")
        print("  1: Pothole Issues")
        print("  2: Illegal Parking Issues")
        print("  3: Broken Road Sign Issues")
        print("  4: Fallen trees")
        print("  5: Littering/Garbage on Public Places")
        print("  6: Vandalism Issues")
        print("  7: Dead Animal Pollution")
        print("  8: Damaged concrete structures")
        print("  9: Damaged Electric wires and poles")
        print()
        
        classes_input = input("Enter class IDs (comma-separated, e.g., '3' or '0,1,3'): ").strip()
        if not classes_input:
            print("No classes specified, using default [3]")
            classes = [3]
        else:
            try:
                classes = [int(x.strip()) for x in classes_input.split(',')]
            except:
                print("Invalid input, using default [3]")
                classes = [3]
        
        checkpoint_path = input("Enter checkpoint path: ").strip()
        if not checkpoint_path:
             print("Checkpoint path required.")
             sys.exit(1)

        conf_input = input("Confidence threshold (0-1, default 0.5): ").strip()
        conf_threshold = float(conf_input) if conf_input else 0.5
        
        custom_testing(classes, checkpoint_path, conf_threshold)
    else:
        print("Invalid choice, running default (Example 1)")
        example_1_single_class()
