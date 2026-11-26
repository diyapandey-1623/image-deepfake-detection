"""
src/seed.py
-----------
Global seed and determinism configuration for deepfake detection pipelines.
Ensures reproducible dataset sampling, tensor initialization, training iterations,
and evaluation inferences across different environments.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> int:
    """
    Sets global deterministic seeds across Python standard library,
    NumPy, PyTorch CPU, and PyTorch CUDA.
    Configures deterministic algorithm behaviors where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms where supported without crashing on unsupported ops
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    return seed


def seed_worker(worker_id: int):
    """Worker initialization function for PyTorch DataLoaders."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
