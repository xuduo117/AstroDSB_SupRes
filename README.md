# AstroDSB_SupRes

Super-resolution of astronomical maps with a Diffusion Schrödinger Bridge trained only on
simulations. This repository holds the model, training and inference code of the 0910 model used in
"Astro-DSB: Physics-Informed Generative Super-Resolution for Astronomical Imaging Surveys"
(D. Xu et al., ApJ, manuscript AAS81737).

* The `v1.0.0` release of this repository is archived on Zenodo (software record).
* The trained weights (epoch 9101), the analysis scripts that produce the paper's figures and
  tables, and the released maps are in a separate Zenodo record,
  [doi:10.5281/zenodo.24013546](https://doi.org/10.5281/zenodo.24013546).

## Layout

```
configs/      the three warm-started stages of the paper's model, SuperRes_dsb.yaml ->
              SuperRes_dsb_0825.yaml -> SuperRes_dsb_0910.yaml (see configs/README.md);
              side_runs/ holds two runs outside that sequence
core/         bridge diffusion, U-Net and model factory
data/         data loader: simulation patches, sky components, beams, resampling, noise
engine/       training loop
utils/        helpers
main.py       training
evaluate.py   held-out (I100) evaluation; writes the predictions to an HDF5 file
crop_maps.py, inference_chunks.py, stitch_maps.py
              inference on a large map: tile, run the model, blend
inference_npy.py
              inference on .npy maps of any shape (sliding window with Hann blending)
ensemble/     multi-draw posterior ensemble on a map region
tools/        input preparation: Planck maps on the ACT grid, the ACT validation split
tests/        checks of the data loader
slurm/        the job scripts used on the Trillium cluster (edit account, modules and paths)
```

## Installation

```
pip install -r requirements.txt
```

Versions are those used for the paper (Python 3.13.2). `tools/` also needs `astropy`, `healpy` and
`pixell`, which are listed separately in `requirements.txt`.

## Weights

Download `AstroDSB_0910_epoch9101_weights.pt` from the Zenodo data record and place it where the
configuration expects the checkpoint, `results/astro_supres_dsb_0910/latest.pt`
(`experiment.base_log_dir` / `experiment.name` in `configs/SuperRes_dsb_0910.yaml`). The file holds
`model_state` (raw weights, used by `inference_chunks.py` and `inference_npy.py`), `ema_state`
(exponential-moving-average weights, used by `evaluate.py` and the ensemble) and `epoch`; it has no
optimizer state, so it is for inference, not for resuming training.

## Usage

Run every command from the repository root.

* Training: `python main.py --config configs/SuperRes_dsb_0910.yaml` (the paper used
  `torchrun --nproc_per_node=4`, see `slurm/train.slurm`). The simulation data paths are set in the
  configuration file.
* Inference on a large map, in three steps:
  1. `python crop_maps.py`: cuts each input map into 256 x 256 px patches at stride 128 and writes
     chunks and `metadata.pkl` (edit the file list at the bottom of the script);
  2. `python inference_chunks.py --config configs/SuperRes_dsb_0910.yaml --data_dir <cropped dir>`:
     runs the model on every chunk (20 network evaluations, raw weights, as for the paper's
     real-data maps);
  3. `python stitch_maps.py`: blends the chunks back into a map with a Hann window.
* Inference on `.npy` maps: `python inference_npy.py --config configs/SuperRes_dsb_0910.yaml
  --data_dir <dir> --out_dir <dir>`.
* Held-out evaluation: `python evaluate.py --config configs/SuperRes_dsb_0910.yaml`.
* Ensemble: `python ensemble/dsb_ensemble_infer.py --code-dir . --config ... --ckpt ...`
  (`--help` lists the options).

Paths to data inside the scripts and configuration files point to the authors' cluster
directories and must be edited.

## Licence

NVIDIA Source Code License for I2SB (`LICENSE`): use for non-commercial research or evaluation
only. Parts of `core/` keep the MIT notice of openai/guided-diffusion (`LICENSE_GUIDED_DIFFUSION`).
`NOTICE.md` lists the adapted files.

## Citation

Please cite the paper and the archived code (`CITATION.cff`).
