"""
Example Testing Scripts for Different Urban Issue Scenarios
============================================================
Shows how to test on different combinations of urban issue classes.
"""

from rcnn import Config, test


def example_1_single_class():
    """Example 1: Test on single class (Broken Road Signs)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Testing on Single Class (Broken Road Signs)")
    print("="*80)
    
    config = Config(urban_issue_classes=[3])  # Class 3: Broken Road Sign Issues
    config.conf_threshold = 0.5  # Adjust as needed
    
    results = test(config, 'checkpoints/best_model.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_2_road_issues():
    """Example 2: Test on all road-related issues"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Testing on Road-Related Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 3])  # Road damage, potholes, signs
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/best_model.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_3_infrastructure():
    """Example 3: Test on infrastructure issues"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Testing on Infrastructure Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[3, 8, 9])  # Signs, Concrete, Electrical
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/best_model.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_4_environmental():
    """Example 4: Test on environmental issues"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Testing on Environmental Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[4, 5, 7])  # Trees, garbage, dead animals
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/best_model.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def example_5_all_classes():
    """Example 5: Test on ALL urban issue classes"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Testing on ALL Classes (Comprehensive Model)")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    config.conf_threshold = 0.5
    
    results = test(config, 'checkpoints/best_model.pth')
    
    print("\n✓ Testing complete!")
    print(f"  mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def custom_testing(class_list, checkpoint_path='checkpoints/best_model.pth', conf_threshold=0.5):
    """Custom testing with specified classes"""
    print("\n" + "="*80)
    print(f"CUSTOM: Testing on Classes {class_list}")
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
    print("RUIDR Testing Examples")
    print("="*80)
    
    # Check if checkpoint exists
    if not os.path.exists('checkpoints/best_model.pth'):
        print("\n⚠ Warning: checkpoints/best_model.pth not found!")
        print("Please train a model first or specify a different checkpoint path.")
        checkpoint_input = input("\nEnter checkpoint path (or press Enter to exit): ").strip()
        if not checkpoint_input or not os.path.exists(checkpoint_input):
            print("Exiting...")
            sys.exit(0)
        checkpoint_path = checkpoint_input
    else:
        checkpoint_path = 'checkpoints/best_model.pth'
    
    print(f"\nUsing checkpoint: {checkpoint_path}")
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
        
        conf_input = input("Confidence threshold (0-1, default 0.5): ").strip()
        conf_threshold = float(conf_input) if conf_input else 0.5
        
        custom_testing(classes, checkpoint_path, conf_threshold)
    else:
        print("Invalid choice, running default (Example 1)")
        example_1_single_class()
    
    print("\n" + "="*80)
    print("Check results:")
    print("  - Metrics: results/evaluation_results.json")
    print("  - Visualizations: results/eval_comparison_*.png")
    print("="*80)

