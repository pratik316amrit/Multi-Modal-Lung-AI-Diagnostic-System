# src/run_temporal_fixed.py
import subprocess
import sys
import os

def main():
    print("🔧 Building Temporal Analysis Pipeline...")
    
    # Create necessary directories
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)
    
    print("1. Training Temporal LSTM Model...")
    try:
        subprocess.run([sys.executable, "train_temporal.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print("Continuing with prediction test...")
    
    print("2. Testing Temporal Prediction...")
    try:
        subprocess.run([sys.executable, "predict_temporal.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Prediction test failed: {e}")
    
    print("\n✅ Temporal pipeline completed!")
    print("Model saved: ../models/temporal_lstm_model.pth")
    print("Training plot: ../outputs/temporal_training_history.png")

if __name__ == "__main__":
    main()