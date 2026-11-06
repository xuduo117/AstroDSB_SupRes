# AstroDSB_SupRes

Code for super-resolving astronomical images with a Diffusion Schrödinger Bridge trained only on
simulations: the model, training and inference code used in "Astro-DSB: Physics-Informed Generative
Super-Resolution for Astronomical Imaging Surveys" (D. Xu et al., ApJ, manuscript AAS81737).
This version (tag `v1.0.0`) is the code of the 0910 model used in the paper.

The `v1.0.0` release of this repository is archived on Zenodo. The trained weights (epoch 9101),
the analysis scripts and the released maps are in a separate Zenodo record; both DOIs are given in
the paper's data-availability statement.

## Contents

* `main.py`: training entry point (`python main.py --config configs/SuperRes_dsb_0910.yaml`).
* `configs/`: configuration files; `SuperRes_dsb_0910.yaml` is the paper's model, the final stage
  of a warm-started lineage (0818, 0825, 0910) described in the paper.
* `core/`: network and bridge diffusion; `engine/`: training loop; `data/`: data loader with the
  degradation model (beams, resampling, noise, contaminants).
* `eval_script/dsb_ensemble_infer.py`: the multi-draw posterior ensemble; `make_val_split.py`,
  `crop_region.py`: the ACT validation split. The analysis scripts that turn model outputs into
  the paper's figures and tables are archived with the weights (data record).
* Inference on a large map, in three steps:
  1. `crop_maps.py`: cuts the input map into 256 x 256 px patches at stride 128 and writes
     chunks plus `metadata.pkl` (edit the file list at the bottom of the script);
  2. `inference_chunks.py --config <yaml> --data_dir <cropped dir>`: runs the model on every chunk
     (20 network evaluations, raw weights, as in the paper's single real-data maps);
  3. `stitch_maps.py`: blends the chunks with a Hann window back into a map.

Paths in the scripts point to the authors' cluster directories and must be edited.

## Licence

NVIDIA Source Code License for I2SB (`LICENSE`): use for non-commercial research or evaluation
only. Parts of `core/` keep the MIT notice of openai/guided-diffusion (`LICENSE_GUIDED_DIFFUSION`).
`NOTICE.md` lists the adapted files.

## Citation

Please cite the paper and the archived code (`CITATION.cff`).
