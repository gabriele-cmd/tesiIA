"""
device = get_device()   →  "cpu"
device = get_device()   →  "cuda:0"  (automatico, zero modifiche)
Con più GPU: device = get_device(gpu_id=1)  →  "cuda:1"
"""

import torch

def get_device(gpu_id: int = 0, verbose: bool = True) -> torch.device:
    """
        gpu_id:  indice della GPU da usare se disponibili più GPU.
                 Ignorato su CPU/MPS.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        if verbose:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            mem_gb   = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            print(f"Device selezionato: GPU — {gpu_name} ({mem_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        if verbose:
            n_threads = torch.get_num_threads()
            print(f"Device selezionato: CPU ({n_threads} thread)")
            print("  → Per usare la GPU: installa PyTorch con supporto CUDA")
            print("    https://pytorch.org/get-started/locally/")

    return device


def move_to_device(*tensors: torch.Tensor, device: torch.device) -> tuple:
    return tuple(t.to(device) for t in tensors)


def to_cpu(*tensors: torch.Tensor) -> tuple:
    """
    Riporta tensori su CPU (es. per salvarli o stamparli).
    Necessario perché numpy non lavora con tensori su GPU.
    """
    return tuple(t.cpu() for t in tensors)


def device_info(device: torch.device) -> dict:
    info = {"device": str(device)}

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["name"]         = props.name
        info["memory_gb"]    = round(props.total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
        info["n_cores"]      = props.multi_processor_count
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"
    else:
        info["name"]     = "CPU"
        info["n_threads"] = torch.get_num_threads()

    return info

"""
# Test rapido

if __name__ == "__main__":
    print("=" * 50)
    print("Test device.py")
    print("=" * 50)

    device = get_device(verbose=True)
    print(f"\nDevice info completa:")
    for k, v in device_info(device).items():
        print(f"  {k}: {v}")

    # Verifica che i tensori si spostino correttamente
    x = torch.randn(4, 4)
    x_dev, = move_to_device(x, device=device)
    assert str(x_dev.device).startswith(device.type), \
        f"Tensore su {x_dev.device}, atteso {device}"
    print(f"\n✓ Tensore spostato correttamente su {x_dev.device}")
"""
