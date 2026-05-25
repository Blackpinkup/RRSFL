# RRSFL: Redundancy-Restricted Shared Feature Learning

This directory implements the RRSFL method on top of the HarmoFL code structure.
The default algorithm follows the paper components:

- Shared Feature Learning (SFL): four encoder-stage client embeddings are projected by local projection heads, aligned with the SFE loss, and used for centroid-aware cosine reweighting.
- Intra-Client Whitening (ICW): local group-wise relaxed whitening regularization suppresses redundant channel correlations.
- Stage-wise server aggregation: each encoder stage is aggregated with its own SFL weights. Projection heads are kept client-local.

## Train

```bash
cd federated
python fed_train.py --log --data camelyon17
python fed_train.py --log --data prostate --batch 16
python fed_train.py --log --data dgdr --batch 16
```

Default optimization follows the paper: Adam, learning rate `1e-4`, betas `(0.9, 0.99)`, weight decay `1e-5`, batch size `16`, one local epoch and one aggregation per round, `100` communication rounds, and `lambda_icw=1e-4`.

## Data

The original HarmoFL Camelyon17 and prostate dataset layout is retained. DGDR supports either:

- `data/DGDR/{SITE}/{split}/{class_name}/*.jpg`
- CSV metadata at `data/DGDR/{SITE}/{split}.csv`, `data/DGDR/{SITE}-{split}.csv`, or `data/DGDR/{SITE}_{split}.csv`

CSV files must contain an image path column named one of `image`, `path`, `img`, `file`, `filename` and a label column named one of `label`, `grade`, `level`, `class`.

## Useful Options

- `--lambda_icw`: ICW balancing weight, default `1e-4`.
- `--icw_groups`: ICW channel group count. `0` uses the number of clients.
- `--sfe_steps`: SFE projection-head optimization steps per round.
- `--disable_icw`, `--disable_sfl`, `--disable_sfe`: ablation switches.
- `--nonnegative_sfl_weights`: clamp cosine aggregation weights to non-negative values.
- `--use_amp_norm`: optionally enable HarmoFL amplitude normalization. It is disabled by default because RRSFL is defined by SFL and ICW.
