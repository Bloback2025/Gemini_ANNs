Python
"""
Date of Creation: September 6, 2026
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\check_tf_gpu_gem.py

Annotations:
- This is a clean rewrite of the legacy check_tf_gpu.py diagnostic script.
- Designed to verify the active Python environment, TensorFlow version, CUDA compilation status, and physical GPU visibility under WSL2.
- Outputs diagnostic details in clean JSON format for quick verification before executing heavy training loops.
"""

import json, sys
import tensorflow as tf

info = {
  "python_executable": sys.executable,
  "tf_version": tf.__version__,
  "built_with_cuda": tf.test.is_built_with_cuda(),
  "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
  "build_info": tf.sysconfig.get_build_info()
}

print(json.dumps(info, indent=2))