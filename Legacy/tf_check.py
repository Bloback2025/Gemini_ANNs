import json
import tensorflow as tf
print(json.dumps({
  "tf": tf.__version__,
  "built_with_cuda": tf.test.is_built_with_cuda(),
  "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]
}, indent=2))
