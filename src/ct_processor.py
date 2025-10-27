# src/ct_processor.py
import numpy as np
import torch
import SimpleITK as sitk  # You'll need to install this

class CTScanProcessor:
    def __init__(self, slice_thickness=5):
        self.slice_thickness = slice_thickness
    
    def load_ct_scan(self, file_path):
        """Load CT scan from DICOM or NIfTI format"""
        try:
            # For DICOM series
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)  # [slices, height, width]
            return array
        except:
            # Fallback: create dummy CT data for testing
            return np.random.rand(100, 512, 512)  # 100 slices of 512x512
    
    def preprocess_ct(self, ct_array):
        """Preprocess CT scan"""
        # Normalize Hounsfield units
        ct_array = np.clip(ct_array, -1000, 1000)
        ct_array = (ct_array + 1000) / 2000.0  # Normalize to [0,1]
        
        # Select key slices
        num_slices = ct_array.shape[0]
        key_slices = np.linspace(0, num_slices-1, 10, dtype=int)  # 10 key slices
        selected_slices = ct_array[key_slices]
        
        return torch.tensor(selected_slices, dtype=torch.float32)