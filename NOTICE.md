# Third-party code in this repository

This repository is distributed under the NVIDIA Source Code License for I2SB (`LICENSE`), because
part of it is adapted from I2SB. The files concerned carry a header saying so.

1. **NVlabs/I2SB** (https://github.com/NVlabs/I2SB), Copyright (c) 2023 NVIDIA Corporation,
   NVIDIA Source Code License for I2SB (`LICENSE`). Adapted here:
   * `core/diffusion.py`: `compute_gaussian_product_coef`, `q_sample`, `p_posterior` and
     `ddpm_sampling` follow `i2sb/diffusion.py`;
   * `utils/util.py`: `unsqueeze_xdim` and `count_parameters` follow `i2sb/util.py`.
   Under Section 3.3 of that licence, the Work and derivative works may be used only
   non-commercially (for research or evaluation).
2. **openai/guided-diffusion** (https://github.com/openai/guided-diffusion), Copyright (c) 2021
   OpenAI, MIT License (`LICENSE_GUIDED_DIFFUSION`). Taken with little change: `core/nn.py`,
   `core/fp16_util.py`; partly: `core/unet.py` (`UNetModel`). These files keep the MIT notice.
3. **L-YeZhu/AstroDSB** (https://github.com/L-YeZhu/AstroDSB; Zhu et al. 2025, NeurIPS), the
   Astro-DSB framework this code extends. It is public without a licence file; its author is a
   co-author of the paper.

The rest of the code is by Duo Xu and is released under the same terms (Section 3.2 of the NVIDIA
licence allows additional terms only if they keep the non-commercial limitation).
