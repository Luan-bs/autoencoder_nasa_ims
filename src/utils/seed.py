"""
src/utils/seed.py

Deterministic seed control for reproducibility across Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds across standard random, NumPy, and PyTorch (if loaded)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

