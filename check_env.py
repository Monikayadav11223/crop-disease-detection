"""check_env.py — Detect installed packages and hardware."""
import sys
print(f"Python: {sys.version}")

packages = [
    ("tensorflow", "tensorflow"),
    ("streamlit", "streamlit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("Pillow", "PIL"),
    ("opencv-python", "cv2"),
    ("reportlab", "reportlab"),
    ("seaborn", "seaborn"),
    ("kaggle", "kaggle"),
]

print("\nPackage availability:")
for label, mod in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "installed")
        print(f"  [OK]  {label}: {ver}")
    except ImportError:
        print(f"  [--]  {label}: NOT INSTALLED")

# GPU check
print("\nGPU / hardware:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  TF GPUs: {gpus if gpus else 'None (CPU only)'}")
except Exception as e:
    print(f"  TF not available: {e}")
