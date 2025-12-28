import torch
import sys
import psutil

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA Available: No")

print(f"MPS Available: {torch.backends.mps.is_available()}")

# Check memory
mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / (1024**3):.2f} GB")
print(f"Available Memory: {mem.available / (1024**3):.2f} GB")

# Test a small generation if possible (requires transformers)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Transformers imported successfully")
except ImportError:
    print("Transformers not found")
