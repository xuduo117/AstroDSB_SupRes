# ---------------------------------------------------------------
# Portions adapted from NVlabs/I2SB (https://github.com/NVlabs/I2SB):
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

def set_seed(seed):
    """
    Ensures reproducibility across different runs and GPUs.
    Moved from original train.py.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # Standard practice for deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataLoaderX(DataLoader):
    """
    High-performance pre-fetching loader.
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def count_parameters(model):
    """Returns total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unsqueeze_xdim(tensor, xdim):
    """
    Helper for tensor broadcasting.
    Ensures scalars or 1D tensors match the spatial dimensions of images.
    """
    for _ in range(len(xdim)):
        tensor = tensor.unsqueeze(-1)
    return tensor