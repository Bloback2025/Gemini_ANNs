from tensorflow import keras
paths = {
  "archive": r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\ho_artifact_outputs\archives\restored_run_20251223_204139\model_best.keras",
  "retrain": r"C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\ho_artifact_outputs\run_retrain_smoke\retrain_patch\retrained_model.keras"
}
for k,p in paths.items():
    try:
        m = keras.models.load_model(p)
        print(f"---- {k} ({p}) ----")
        print("input_shape:", m.input_shape)
    except Exception as e:
        print(f"Could not load {k} at {p}: {e}")



