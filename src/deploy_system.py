# src/deploy_system.py
import argparse
import json
from complete_pipeline import CompleteLungAISystem

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Lung AI System")
    parser.add_argument('--image', type=str, required=True, help='Path to chest X-ray/CT')
    parser.add_argument('--clinical', type=str, help='JSON file with clinical data')
    parser.add_argument('--previous', nargs='+', help='Previous scans for temporal analysis')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CompleteLungAISystem()
    
    # Load clinical data if provided
    clinical_data = None
    if args.clinical:
        with open(args.clinical, 'r') as f:
            clinical_data = json.load(f)
    
    # Run complete analysis
    results = system.predict_complete(
        image_path=args.image,
        clinical_data=clinical_data,
        previous_scans=args.previous
    )
    
    # Print results
    print("\n" + "="*60)
    print("🩺 MULTI-MODAL LUNG AI DIAGNOSIS REPORT")
    print("="*60)
    
    print("\n📊 DISEASE PREDICTIONS:")
    for disease, prob in results['predictions'].items():
        status = "🟢 LOW" if prob < 0.3 else "🟡 MODERATE" if prob < 0.7 else "🔴 HIGH"
        print(f"   {disease:<15}: {prob:.3f} ({status})")
    
    print(f"\n🔍 EXPLANATION:")
    print(f"   Image contribution: {results['explanation']['image_contribution']:.2%}")
    print(f"   Clinical contribution: {results['explanation']['clinical_contribution']:.2%}")
    if results['explanation']['key_factors']:
        print("   Key clinical factors:")
        for factor in results['explanation']['key_factors']:
            print(f"     - {factor}")
    
    if results['temporal_analysis']:
        print(f"\n🕒 TEMPORAL ANALYSIS:")
        for trend in results['temporal_analysis']['trends']:
            print(f"   {trend}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()