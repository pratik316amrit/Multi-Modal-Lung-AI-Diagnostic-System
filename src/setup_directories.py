# src/setup_directories.py
import os
from pathlib import Path

def setup_project_structure():
    """Create all necessary directories for the project"""
    
    base_dirs = [
        "../data/raw/ct_scans",
        "../data/processed/ct_slices",
        "../data/clinical",
        "../data/splits",
        "../models",
        "../outputs/gradcam",
        "../outputs/masks", 
        "../outputs/predictions",
        "../outputs/temporal",
        "../outputs/multi_modal"
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create placeholder files if needed
    placeholder_files = [
        "../data/raw/ct_scans/README.md",
        "../data/processed/ct_slices/README.md"
    ]
    
    for file_path in placeholder_files:
        if not Path(file_path).exists():
            with open(file_path, 'w') as f:
                f.write("# Placeholder for CT scan data\n")
            print(f"📝 Created: {file_path}")
    
    print("\n🎯 Project structure setup complete!")

if __name__ == "__main__":
    setup_project_structure()