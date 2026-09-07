import subprocess, sys, textwrap

ENV = "tf_cf_py39"
PY = sys.executable

def run(cmd, capture=False):
    print("\n$ " + " ".join(cmd))
    if capture:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd)

print("Attempting to install cuDNN from NVIDIA channel (conda-managed)...")
res = run(["mamba", "install", "-n", ENV, "-c", "nvidia", "cudnn", "-y"], capture=True)
print(res.stdout)
if res.returncode != 0:
    print("\n=== mamba install failed ===")
    print("Running mamba search -c nvidia cudnn to list available builds...")
    res2 = run(["mamba", "search", "-c", "nvidia", "cudnn"], capture=True)
    print(res2.stdout)
    print(f"\nIf you see a suitable package, install it with:\n  mamba install -n {ENV} -c nvidia <package-name> -y")
    sys.exit(1)

print("\ncuDNN installed via NVIDIA channel successfully.")
print("\nReinstalling TensorFlow with pinned ABI packages...")
run([PY, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-intel", "tensorflow-estimator"])
run([PY, "-m", "pip", "install", "--no-cache-dir", "numpy<1.25", "protobuf<4.24", "wheel", "setuptools"])
run([PY, "-m", "pip", "install", "--no-cache-dir", "tensorflow==2.11.0"])

check_code = textwrap.dedent("""
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
""")
with open("check_tf_gpu.py", "w", encoding="utf8") as f:
    f.write(check_code)

print("\nRunning GPU check script...")
run([PY, "check_tf_gpu.py"])
print("\nListing DLLs in env Library\\bin (if available):")
run(["powershell", "-Command", "Get-ChildItem -Path \"$env:CONDA_PREFIX\\Library\\bin\" -Filter \"*cudnn*.dll\" -ErrorAction SilentlyContinue"])
run(["powershell", "-Command", "Get-ChildItem -Path \"$env:CONDA_PREFIX\\Library\\bin\" -Filter \"cudart*.dll\" -ErrorAction SilentlyContinue"])
