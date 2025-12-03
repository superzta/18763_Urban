"""
FCOS Training Examples
======================
Shows how to train FCOS on different combinations of urban issue classes.
"""

from rcnn import Config
from fcos import train

def example_1_single_class():
    """Example 1: Train on single class (Broken Road Signs)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Training FCOS on Single Class (Broken Road Signs)")
    print("="*80)
    
    config = Config(urban_issue_classes=[3])  # Class 3: Broken Road Sign Issues
    config.num_epochs = 2
    config.batch_size = 4
    config.learning_rate = 0.001  # FCOS often needs lower LR
    
    model = train(config)
    print("\n✓ Training complete!")


def example_2_road_issues():
    """Example 2: Train on all road-related issues"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Training FCOS on Road-Related Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 3])  # Road damage, potholes, signs
    config.num_epochs = 20
    config.batch_size = 4
    config.learning_rate = 0.001
    
    model = train(config)
    print("\n✓ Training complete!")


def example_5_all_classes():
    """Example 5: Train on ALL urban issue classes"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Training FCOS on ALL Classes (Comprehensive Model)")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    config.num_epochs = 50
    config.batch_size = 4
    config.learning_rate = 0.001
    
    model = train(config)
    print("\n✓ Training complete!")


if __name__ == '__main__':
    print("="*80)
    print("FCOS Training Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Single class (Broken Road Signs)")
    print("  2. Road-related issues (Damaged roads, Potholes, Signs)")
    print("  5. ALL classes (Comprehensive model)")
    print()
    
    choice = input("Select example (1, 2, 5) or Enter for default (1): ").strip()
    
    if choice == '1' or choice == '':
        example_1_single_class()
    elif choice == '2':
        example_2_road_issues()
    elif choice == '5':
        example_5_all_classes()
    else:
        print("Invalid choice, running default (Example 1)")
        example_1_single_class()
