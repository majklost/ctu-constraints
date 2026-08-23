# Upcoming features checklist

Checklist of features to be implemented in this repo

# 0. Revision

- [ ] revise if into metrics and losses I pass probabilities or raw logits (mainly from UNET prediction), maybe unify or offer clear contract
- [ ] Add to loss term a fieldRegularization term for penalizing norm of gradient of deformation field - i.e. penalize large spatial gradients
- [ ] Add logging of determinant of jacobian, idea of tracking:

```python
per batch:
  jac = compute_jacobian_det(displacement_field)  # per-pixel det J, shape [B, H, W]
  neg_frac_pixels = (jac <= 0).float().mean(dim=[1,2])       # per-sample pixel fraction
  sample_has_violation = (jac <= 0).any(dim=[1,2])           # per-sample bool
  min_det = jac.amin(dim=[1,2])                              # per-sample worst case

epoch aggregate (log all three):
  mean(neg_frac_pixels)          # overall severity
  mean(sample_has_violation)     # how widespread
  mean(min_det), or 1st percentile of min_det across samples  # worst-case tracking
```

- [ ] analyze if it is possible to log slurm job ID to wandb

# 1. Model weight saving

- [ ] verify if saving torch.save is enough if we suppose that training wont be resumed
- [ ] create a weight saving system, probably `get_weights_folder` similar to `get_experiment_folder`, weights should live in the `synced` folder but not be synced, instead proposing the structure
  - each experiment has its own folder `ex5`,...
  - each file there has its own folder
  - inside for each run there is a folder
  - in folder there are `weights.pth` and `metadata.json` , metadata containains name of wandb run, time of creation, e.g. a slurm job ID if available, weights size
  - thanks two mutagen two-way-sync it will sync the folder structure with metadata
  - it is now easy to specify weights_folder to load the weights without costly syncing them

# 2.Harder dataset creation

- [ ] - deprecate rigid/affine dataset creation
- [ ] - implement rigid-on-demand
  - a script, where for data in dataset, we generate random rotation and translation that create the alignment, this is stored alongside the rest of data in .npy file
    - [ ] implement check and rejection sampling so it does guarantees no excee of boundaries
    - [ ] in `CachedArtificialDataset` support optional loading of these parameters,  transform also the SDF representation respectively... Use convenient interpolation (e.g. for masks)
- [ ] - plot different magnitudes of deformation, get to know how to affect to get field with small agressive and wide deformations
- [ ] - add an option to generate more plaques than two, option to specify plaque sizes
      - parametric generation of plaque
- [ ] - add option to create low frequency noise that has similar structure as plaques (would mimic echogenity of artery walls)
- [ ] - add option for black rectangles, mimicing ultrasound suppression created by bone

## Experiment details
- this will lead to 9 different datasets (each about 2000 trn samples and 100 val)
  - two plaques, same size, different angle (easy/hard mode)
    - do hard after results of easy (maybe easy becomes the hard XD)
  - two plaques, different size, same angle
  - two plaques - different both (easy-easy, hard-hard)
  - fake plaques
    - [ ] this should be low-frequency noise that will confuse UNET, think how to do it
  - black rectangles
  - different size and angle + fake plaques
- keep 6 methods for inspection
  - UNET
  - One-side SDF (seg One-Side)
  - One-side SDF (seg BCE)
  - DSDF MSE (seg BCE)
  - One-side SDFSquared (seg BCE)
  - Centroid (seg BCE)

# 3. Multiple template deformation
- [ ] analyze how to performs passing of multiple templates at once (if paralelize or not)
- [ ] implement 3 different template positions and sizes (possibly overlap with point 2. - e.g. 4 or 6 templates)
  - after results from point 2 decide the correct angle and sizes
  - check if correct template gets selected

# 4. Connection of external dataset
- [ ] connect CCA dataset, evaluate promising runs
  - [ ] connect CSV dataset (if CCA successful), decide if convenient template can be deduced
- [ ] connect ACDC dataset, evaluate promising runs


# 5. Branch architecture
- will be specified later

# 6. Anatomically constrained neural network implementation and TEDS-NET implementation
- will be specified later
