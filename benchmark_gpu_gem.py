Python
"""
Date of Creation: September 6, 2026
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\benchmark_gpu_gem.py

Annotations:
- This is a clean rewrite of the legacy benchmark script, renamed to prevent version confusion.
- Updated for the Gemini_ANNs project root directory.
- Designed to verify TensorFlow GPU device detection, enforce memory growth safely under WSL2, and measure hardware throughput and JIT compilation overhead for high-performance ANN workloads.
"""

import time
import subprocess
import tensorflow as tf
import numpy as np

def get_gpu_vram_usage():
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split(',')
        return {"used_mb": int(output[0]), "total_mb": int(output[1])}
    except Exception:
        return {"used_mb": 0, "total_mb": 0}

print("=" * 60)
print("             WSL2 TENSORFLOW GPU BENCHMARK SUITE          ")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("❌ ERROR: No GPU detected by TensorFlow. Exiting bench.")
    exit(1)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"✅ Target GPU Detected: {gpus[0].name}")
print(f"📌 TF Version: {tf.__version__} | NumPy Version: {np.__version__}")
initial_vram = get_gpu_vram_usage()
print(f"📊 Baseline VRAM Footprint: {initial_vram['used_mb']}MB / {initial_vram['total_mb']}MB")
print("-" * 60)

print("🚀 Phase 1: Measuring JIT Compilation & Warmup Overhead...")
matrix_size = 8192
a = tf.random.normal((matrix_size, matrix_size))
b = tf.random.normal((matrix_size, matrix_size))

start_cold = time.perf_counter()
c_cold = tf.matmul(a, b)
_ = c_cold.numpy()
end_cold = time.perf_counter()
jit_time = end_cold - start_cold
print(f"⏱️  Cold Run (JIT Compilation Included): {jit_time:.4f} seconds")

start_hot = time.perf_counter()
c_hot = tf.matmul(a, b)
_ = c_hot.numpy()
end_hot = time.perf_counter()
hot_time = end_hot - start_hot
print(f"⏱️  Hot Run (Pure Hardware Execution):  {hot_time:.4f} seconds")
print(f"⚡ JIT Compilation Penalty:             {jit_time - hot_time:.4f} seconds")
print("-" * 60)

print("🏋️ Phase 2: Simulating High-Throughput ANN Training Loop...")
steps = 200
batch_size = 256
input_dim = 1000
output_dim = 10

x_train = tf.random.normal((batch_size * steps, input_dim))
y_train = tf.random.uniform((batch_size * steps, output_dim))
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(output_dim)
])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

step_times = []
print(f"🔄 Executing {steps} steps across dense topology...")

start_loop = time.perf_counter()
for step, (x_batch, y_batch) in enumerate(dataset):
    step_start = time.perf_counter()
    loss = train_step(x_batch, y_batch)
    _ = loss.numpy()
    step_end = time.perf_counter()
    if step > 0:
        step_times.append(step_end - step_start)

end_loop = time.perf_counter()

total_loop_time = end_loop - start_loop
avg_step_time = np.mean(step_times)
throughput = batch_size / avg_step_time

print("-" * 60)
print("📊 PROCESSED PERFORMANCE METRICS:")
print(f"⏱️  Total Loop Wall-Clock Time: {total_loop_time:.2f} seconds")
print(f"⏱️  Average Step Processing Time: {avg_step_time * 1000:.2f} ms")
print(f"🔥 System Throughput Capacity:   {throughput:.2f} samples/second")
final_vram = get_gpu_vram_usage()
print(f"📊 Peak Active VRAM Allocation:  {final_vram['used_mb']}MB / {final_vram['total_mb']}MB")
print("=" * 60)