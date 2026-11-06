# Configurations

The model of the paper (weights at epoch 9101) is the end of three warm-started stages. Each stage
restored the weights, optimizer and schedule of the previous one; the sequence below is the one
recorded in the training logs.

| Stage | Config | Run name | Epochs |
|---|---|---|---|
| base run, from scratch | `SuperRes_dsb.yaml` | `astro_supres_dsb` | 0–4101 |
| second stage | `SuperRes_dsb_0825.yaml` | `astro_supres_dsb_0825` | 4102–6101 |
| final stage (paper) | `SuperRes_dsb_0910.yaml` | `astro_supres_dsb_0910` | 6102–9101 |

The second stage started from the base run's epoch-4101 checkpoint. The comment at the top of
`SuperRes_dsb_0825.yaml` says it was warm-started from the 0818 weights; the training logs show
otherwise. `side_runs/` holds two runs that are not part of this sequence, 0818 and 0901; the
comments in `SuperRes_dsb_0910.yaml` describe how 0910 differs from 0901. A stage is resumed by copying the previous stage's `latest.pt` into
`results/<run name>/` before the first launch, as described in the comments of each file.
