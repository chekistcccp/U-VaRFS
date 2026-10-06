from contextlib import contextmanager

import torch


@contextmanager
def precise_matmul():
    """Full FP32 matmul for fitted objectives; restore detector/backbone policy."""
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
