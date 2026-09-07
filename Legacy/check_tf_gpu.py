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
