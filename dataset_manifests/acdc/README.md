- 1,902 source slices from 100 patients
- 1,798 slices have `annular_myocardium=true`
- `trn.csv`, `val.csv`, and `test.csv` are a deterministic (seed 42),
  patient-disjoint 70/20/10 split of the annular slices
- images do not all have the same size: there are 29 spatial sizes; the most
  common is `(256, 216)` (506 slices)
- 4 source classes: background, right ventricle, myocardium, left ventricle
- UNet training uses a binary target (`myocardium == 2`) and resizes images and
  masks to `(256, 256)` for batching
