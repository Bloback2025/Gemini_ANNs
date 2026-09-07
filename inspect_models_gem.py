"""
Date of Creation: September 6, 2026, 7:03 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\inspect_models_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\inspect_models.py

Annotations:
- This is a clean rewrite of the model inspection and diagnostic script, updated for the Gemini_ANNs project root directory[cite: 16].
- Safely iterates through legacy model artifact paths to test Keras model loading compatibility and inspect expected input shapes[cite: 16].
"""

from tensorflow import keras

# Updated paths for inspection reference if needed, or pointing to local artifacts
paths = {
  "archive": r"C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\artifacts\archives\restored_run_20251223_204139\model_best.keras",
  "retrain": r"C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\artifacts\run_retrain_smoke\retrain_patch\retrained_model.keras"
}

for k, p in paths.items():
    try:
        m = keras.models.load_model(p)
        print(f"---- {k} ({p}) ----")
        print("input_shape:", m.input_shape)
    except Exception as e:
        print(f"Could not load {k} at {p}: {e}")