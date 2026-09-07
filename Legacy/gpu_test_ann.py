import tensorflow as tf
import numpy as np
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
print("Python:", __import__('sys').executable)
print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)

# tiny ANN on random data to force GPU kernels
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
