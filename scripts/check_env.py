"""Проверка окружения для обучения U-Net."""
import sys

print("Python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
except Exception as e:
    print("torch ERROR:", e)

for pkg in ["segmentation_models_pytorch", "albumentations", "cv2", "numpy", "pandas"]:
    try:
        m = __import__("cv2" if pkg == "cv2" else pkg)
        print(pkg, "OK", getattr(m, "__version__", ""))
    except Exception as e:
        print(pkg, "MISSING", e)
