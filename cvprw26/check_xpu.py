import torch
try:
    import intel_extension_for_pytorch
    print("IPEX imported")
except ImportError:
    print("IPEX not found")

print(f"PyTorch version: {torch.__version__}")
if hasattr(torch, "xpu"):
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"XPU device name: {torch.xpu.get_device_name(0)}")
else:
    print("XPU not supported in this torch version")
