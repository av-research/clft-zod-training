import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {gpu_name} (CUDA Capability {capability[0]}.{capability[1]})")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
    # Run a simple CUDA operation to verify functionality
    print("\nRunning a simple CUDA test operation...")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"Test tensor on CUDA: {z.device}")
    print("CUDA test successful!")
else:
    print("No CUDA devices available. Check your NVIDIA drivers and installation.")