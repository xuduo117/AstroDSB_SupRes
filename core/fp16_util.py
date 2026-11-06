# ---------------------------------------------------------------
# Adapted from https://github.com/openai/guided-diffusion
# (guided_diffusion/fp16_util.py), Copyright (c) 2021 OpenAI, MIT License.
# The license for the original version of this file can be found in
# LICENSE_GUIDED_DIFFUSION.
# ---------------------------------------------------------------

import torch as th
import torch.nn as nn

def convert_module_to_f16(l):
    """
    Convert primitive modules (Conv layers) to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    """
    Convert primitive modules back to float32.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()