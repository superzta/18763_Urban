"""
Installation Verification Script
=================================
Run this script to verify that all dependencies are correctly installed.

Usage:
    python verify_installation.py
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} - MISSING ({str(e)})")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ {'CUDA':20s} - Available (GPU: {torch.cuda.get_device_name(0)})")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print(f"⚠ {'CUDA':20s} - Not available (will use CPU)")
            return False
    except Exception as e:
        print(f"✗ {'CUDA':20s} - Error checking CUDA ({str(e)})")
        return False


def check_pytorch_version():
    """Check PyTorch version."""
    try:
        import torch
        version = torch.__version__
        print(f"  PyTorch version: {version}")
        
        # Check if version is sufficient
        major, minor = map(int, version.split('.')[:2])
        if major >= 2 or (major == 1 and minor >= 12):
            return True
        else:
            print(f"  ⚠ Warning: PyTorch {version} detected. Recommend >= 2.0.0")
            return True
    except Exception as e:
        print(f"  ✗ Error checking PyTorch version: {str(e)}")
        return False


def check_dataset():
    """Check if dataset exists."""
    import os
    
    dataset_path = "data/DamagedRoadSigns/DamagedRoadSigns"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(dataset_path):
        print(f"✓ {'Dataset':20s} - Found at {dataset_path}")
        
        # Check for required subdirectories
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_path, split)
            images_path = os.path.join(split_path, 'images')
            labels_path = os.path.join(split_path, 'labels')
            
            if os.path.exists(images_path) and os.path.exists(labels_path):
                num_images = len([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                num_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
                print(f"  {split:8s}: {num_images} images, {num_labels} labels")
            else:
                print(f"  ⚠ {split:8s}: Missing images or labels directory")
        
        if os.path.exists(data_yaml):
            print(f"  data.yaml: Found")
        else:
            print(f"  ⚠ data.yaml: Not found")
        
        return True
    else:
        print(f"✗ {'Dataset':20s} - Not found at {dataset_path}")
        print(f"  Please download and extract the dataset to this location")
        return False


def main():
    """Main verification function."""
    print("=" * 80)
    print("Installation Verification")
    print("=" * 80)
    print()
    
    print("Checking Python version...")
    print(f"  Python {sys.version}")
    if sys.version_info < (3, 8):
        print("  ⚠ Warning: Python 3.8+ is recommended")
    print()
    
    print("Checking required packages...")
    all_ok = True
    
    # Core packages
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('cv2', 'opencv-python'),
    ]
    
    for module, package in packages:
        if not check_import(module, package):
            all_ok = False
    
    print()
    print("Checking PyTorch and CUDA...")
    check_pytorch_version()
    check_cuda()
    
    print()
    print("Checking dataset...")
    check_dataset()
    
    print()
    print("=" * 80)
    if all_ok:
        print("✓ All required packages are installed!")
        print("  You can now run: python rcnn.py --mode train")
    else:
        print("✗ Some packages are missing.")
        print("  Please run: pip install -r requirements.txt")
    print("=" * 80)


if __name__ == '__main__':
    main()

