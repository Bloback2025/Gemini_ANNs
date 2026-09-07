"""
Date of Creation: September 6, 2026, 6:58 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\gpu_test_ann_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\gpu_test_ann.py

Annotations:
- This is a clean rewrite of the quick GPU diagnostic and smoke-test script, migrated to the Gemini_ANNs root structure[cite: 14].
- Verifies physical GPU device detection, active Python executable, CUDA build availability[cite: 14], and logs device placement[cite: 14].
- Fits a tiny synthetic ANN model on random data for 1 epoch to actively exercise and force hardware GPU kernels[cite: 14].
"""

import tensorflow as tf
import numpy as np

# Safely enable memory growth if physical GPU is found
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU Memory Growth Enabled Successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not set memory growth: {e}")

print("Python:", __import__('sys').executable)
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", gpus)

# Enable device placement logging to confirm hardware execution
tf.debugging.set_log_device_placement(True)

# Tiny synthetic ANN on random data to force GPU kernels
x = np.random.rand(1024, 32).astype('float32')
y = np.random.randint(0, 2, size=(1024, 1)).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting 1-epoch training to exercise GPU...")
model.fit(x, y, epochs=1, batch_size=128, verbose=2)
print("Done")