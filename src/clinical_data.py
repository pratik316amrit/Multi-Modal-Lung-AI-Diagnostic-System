# src/clinical_data.py
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path

class ClinicalDataGenerator:
    def __init__(self):
        self.clinical_features = [
            'age', 'gender', 'smoking_history', 'oxygen_saturation',
            'fever', 'cough', 'shortness_of_breath', 'chest_pain'
        ]
        
        # Create directories if they don't exist
        self.clinical_dir = Path("../data/clinical")
        self.clinical_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_for_patients(self, patient_ids, csv_file):
        """Generate synthetic clinical data for patients"""
        df = pd.read_csv(csv_file)
        
        clinical_data = {}
        for patient_id in patient_ids:
            # Extract from image ID format: '00000001_001.png'
            pid = patient_id.split('_')[0] if '_' in patient_id else patient_id
            
            # Generate realistic clinical data based on possible conditions
            clinical_data[patient_id] = {
                'age': np.random.randint(20, 90),
                'gender': np.random.choice([0, 1]),  # 0: female, 1: male
                'smoking_history': np.random.choice([0, 1, 2]),  # 0: never, 1: former, 2: current
                'oxygen_saturation': np.random.uniform(85.0, 99.0),
                'fever': np.random.choice([0, 1]),
                'cough': np.random.choice([0, 1]),
                'shortness_of_breath': np.random.choice([0, 1]),
                'chest_pain': np.random.choice([0, 1])
            }
        
        # Save clinical data
        clinical_df = pd.DataFrame.from_dict(clinical_data, orient='index')
        clinical_df.index.name = 'patient_id'
        
        # Reset index to make patient_id a column
        clinical_df_reset = clinical_df.reset_index()
        
        clinical_path = self.clinical_dir / "clinical_data.csv"
        clinical_df_reset.to_csv(clinical_path, index=False)
        
        print(f"✅ Clinical data saved to: {clinical_path}")
        print(f"📊 Generated clinical data for {len(clinical_data)} patients")
        
        return clinical_df_reset

    def normalize_clinical_features(self, clinical_tensor):
        """Normalize clinical features"""
        # Age: normalize to [0,1]
        clinical_tensor[:, 0] = (clinical_tensor[:, 0] - 20) / 70.0
        # Oxygen saturation: normalize to [0,1]
        clinical_tensor[:, 3] = (clinical_tensor[:, 3] - 85.0) / 15.0
        # Binary features already in [0,1]
        return clinical_tensor

    def create_clinical_json_template(self):
        """Create a JSON template for clinical data input"""
        template = {
            "age": 45,
            "gender": 1,
            "smoking_history": 1,
            "oxygen_saturation": 95.0,
            "fever": 0,
            "cough": 1,
            "shortness_of_breath": 0,
            "chest_pain": 0
        }
        
        template_path = self.clinical_dir / "clinical_template.json"
        import json
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"📝 Clinical template saved to: {template_path}")

# Usage
if __name__ == "__main__":
    print("🩺 Generating Clinical Data...")
    
    generator = ClinicalDataGenerator()
    
    # Get patient IDs from your existing data
    try:
        df = pd.read_csv("../data/splits/train.csv")
        patient_ids = df['image_id'].tolist()
        
        print(f"Found {len(patient_ids)} patients in training data")
        
        # Generate clinical data
        clinical_df = generator.generate_for_patients(patient_ids[:1000], "../data/splits/train.csv")  # Limit to 1000 for speed
        
        # Create template
        generator.create_clinical_json_template()
        
        # Show sample of generated data
        print("\n📋 Sample Clinical Data:")
        print(clinical_df.head(10))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your data splits exist in ../data/splits/")