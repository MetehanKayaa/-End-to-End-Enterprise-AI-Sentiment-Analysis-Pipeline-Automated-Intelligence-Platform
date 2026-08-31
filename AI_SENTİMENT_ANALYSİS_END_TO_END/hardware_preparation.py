import torch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


def gpu_preparation() -> bool:
    """
    Checks GPU availability and logs system capability.
    Returns True if a CUDA-compatible GPU is found, False otherwise.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU detected: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
        return True
    
    print("WARNING: NO GPU detected. Performance will be slow.")
    return False


if __name__ == "__main__":
    gpu_preparation()aimport torch


def gpu_preparation() -> str:
    """
    Detects the best available hardware accelerator across Windows/Linux (CUDA),
    macOS Apple Silicon (MPS), or falls back to CPU.
    Returns: 'cuda', 'mps', or 'cpu'.
    """
    # 1. Check for NVIDIA CUDA (Windows / Linux)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA GPU detected: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
        return "cuda"

    # 2. Check for Apple Silicon Metal MPS (macOS M1/M2/M3/M4)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) detected. Metal acceleration active.")
        return "mps"

    # 3. Fallback to CPU
    else:
        print("WARNING: No GPU or MPS accelerator detected. Performance will be slow on CPU.")
        return "cpu"


def is_accelerator_available() -> bool:
    """
    Convenience helper that returns True if either CUDA or MPS is available.
    """
    return gpu_preparation() in ["cuda", "mps"]


if __name__ == "__main__":
    active_device = gpu_preparation()
    print(f"Runtime compute target: {active_device.upper()}")