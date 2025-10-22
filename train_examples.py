"""
Example Training Scripts for Different Urban Issue Scenarios
=============================================================
Shows how to train on different combinations of urban issue classes.
"""

from rcnn import Config, train, test

def example_1_single_class():
    """Example 1: Train on single class (Broken Road Signs)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Training on Single Class (Broken Road Signs)")
    print("="*80)
    
    config = Config(urban_issue_classes=[3])  # Class 3: Broken Road Sign Issues
    config.num_epochs = 10
    config.batch_size = 4
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")


def example_2_road_issues():
    """Example 2: Train on all road-related issues"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Training on Road-Related Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 3])  # Road damage, potholes, signs
    config.num_epochs = 20
    config.batch_size = 4
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")


def example_3_infrastructure():
    """Example 3: Train on infrastructure issues"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Training on Infrastructure Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[8, 9])  # Concrete + electrical
    config.num_epochs = 15
    config.batch_size = 4
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")


def example_4_environmental():
    """Example 4: Train on environmental issues"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Training on Environmental Issues")
    print("="*80)
    
    config = Config(urban_issue_classes=[4, 5, 7])  # Trees, garbage, dead animals
    config.num_epochs = 15
    config.batch_size = 4
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")


def example_5_all_classes():
    """Example 5: Train on ALL urban issue classes"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Training on ALL Classes (Comprehensive Model)")
    print("="*80)
    
    config = Config(urban_issue_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    config.num_epochs = 50
    config.batch_size = 8  # Larger batch for more classes
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")


def custom_training(class_list, epochs=20):
    """Custom training with specified classes"""
    print("\n" + "="*80)
    print(f"CUSTOM: Training on Classes {class_list}")
    print("="*80)
    
    config = Config(urban_issue_classes=class_list)
    config.num_epochs = epochs
    config.batch_size = 4
    config.learning_rate = 0.005
    
    model = train(config)
    print("\n✓ Training complete!")
    return model


if __name__ == '__main__':
    import sys
    
    print("="*80)
    print("RUIDR Training Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Single class (Broken Road Signs)")
    print("  2. Road-related issues (Damaged roads, Potholes, Signs)")
    print("  3. Infrastructure (Concrete structures, Electrical poles)")
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
        print("Available classes:")
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

        classes_input = input("Classes: ").strip()
        classes = [int(x.strip()) for x in classes_input.split(',')]
        
        epochs_input = input("Number of epochs (default 20): ").strip()
        epochs = int(epochs_input) if epochs_input else 20
        
        custom_training(classes, epochs)
    else:
        print("Invalid choice, running default (Example 1)")
        example_1_single_class()

