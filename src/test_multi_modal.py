# src/test_multi_modal.py
import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import torch
        import pandas as pd
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score
        
        print("✅ All core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_structure():
    """Test if project structure is correct"""
    required_dirs = [
        "../data/processed",
        "../data/splits", 
        "../data/clinical",
        "../models",
        "../outputs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_exist = False
    
    return all_exist

def test_data_files():
    """Test if required data files exist"""
    required_files = [
        "../data/splits/train.csv",
        "../data/splits/val.csv", 
        "../data/processed/00000002_000.png"  # Sample image
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    print("🔧 Testing Multi-Modal System Setup...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    print("\n" + "-" * 50)
    
    # Test 2: Directory structure
    if test_data_structure():
        tests_passed += 1
    
    print("\n" + "-" * 50)
    
    # Test 3: Data files
    if test_data_files():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready to proceed.")
        print("\nNext steps:")
        print("1. python setup_directories.py")
        print("2. python clinical_data.py") 
        print("3. python complete_pipeline.py")
    else:
        print("❌ Some tests failed. Please check the setup.")

if __name__ == "__main__":
    main()