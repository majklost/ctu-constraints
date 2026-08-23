# Upcoming work

Implementation and experiment checklist. The immediate work is ordered so that
correctness and reproducibility changes precede dataset generation and the main
experiment matrix.

## 0. Correctness and reproducibility

### Loss and metric input contracts

- [x] Verify that every term in `loss_terms.py` and `metric_terms.py` follows the
  existing `LossInput` and `MetricInput` contracts.
  - `segmentation_logits` contains raw UNet logits.
  - Segmentation cross-entropy receives raw logits.
  - Losses that operate on soft masks apply `softmax` to segmentation logits.
  - Segmentation metrics discretize logits with `argmax`.
  - `warped_template` is a soft class mask produced by interpolation, not logits;
    registration terms and metrics must treat it accordingly.

### Deformation-field regularity

- [x] Add a diffusion regularization loss term that penalizes spatial gradients
  of the deformation field.
  - Reuse the conventions and machinery already used by VoxelMorph, including
    the choice of field representation, measurement units, and reduction.
      - make use of `neurite` package if convenient... may look to implementation at `notebooks/ex1/voxelmorph_test_MNIST.ipynb` for inspiration
  - Expose the regularization weight in experiment configuration.
  - Log both the unweighted term and its weighted contribution to total loss.
- [x] Add epoch-level Jacobian-determinant diagnostics for the final deformation:
  - mean fraction of pixels with `det(J) <= 0`;
  - fraction of samples containing at least one pixel with `det(J) <= 0`;
  - mean of the per-sample minimum determinant;
  - first percentile of the per-sample minimum determinant.
  - Follow the existing VoxelMorph field and spatial-transform conventions.

### Naming and run metadata

- [ ] Decide whether to rename the domain-level `affine` terminology to `rigid`
  now or retain it for compatibility with existing datasets and experiments.
  - The implemented transform is rotation plus translation and is therefore
    rigid.
  - Keep PyTorch/API concepts such as `affine_matrix`, `affine_grid`, and their
    low-level helpers named affine.
  - If renamed, preserve compatibility for existing cached dataset layouts and
    old CLI arguments where practical.
- [ ] Add Slurm and source metadata to W&B runs:
  - `SLURM_JOB_ID`, when available;
  - Git commit hash and whether the worktree was dirty;
  - experiment filename and command-line arguments.
  - Verify which Git metadata W&B already records automatically, but store the
    commit explicitly so it is also available to local weight metadata.

## 1. Model weight saving

- [ ] Implement inference-weight saving using `model_state_dict`; training does
  not need to be resumable.
- [ ] Select the saved checkpoint using the best validation registration metric.
  Fall back to the best validation segmentation metric when registration is not
  available, as for the UNet-only baseline.
- [ ] Add a `get_weights_folder` helper analogous to
  `get_experiment_folder`, with the structure:

  ```text
  synced/weights/<experiment>/<filename>/<run-id>/
      weights.pth
      metadata.json
  ```

- [ ] Store at least the following in `metadata.json`:
  - W&B run name, run ID, and URL;
  - creation time;
  - experiment filename and command-line arguments;
  - Git commit hash and dirty-worktree status;
  - Slurm job ID, when available;
  - checkpoint-selection metric, value, and epoch;
  - model construction/configuration information needed to load the state dict;
  - weight-file size.
- [ ] Configure Mutagen so the folder hierarchy and metadata are synchronized,
  while `weights.pth` files remain local and are not synchronized.
- [ ] Add a round-trip smoke test: save weights, construct a fresh model, load
  the state dict, and verify identical outputs on a fixed input.

## 2. Harder artificial datasets

Keep the existing rigid/affine dataset creation path for compatibility. New
experiments may use new scripts instead of extending
`create_artificial_dataset.py`.

### Runtime rigid transformation

- [ ] Create a script that samples and stores one rigid transform per sample as
  `(angle, dx, dy)` in an `.npy` file with shape `[N, 3]`.
- [ ] Use rejection sampling to guarantee that transformed foreground classes
  are neither lost outside the image nor touching the image boundary.
  - Define a finite maximum number of attempts and report infeasible parameter
    bounds clearly.
- [ ] Extend `CachedArtificialDataset` to load the optional rigid parameters and
  apply the transform at sample construction time to:
  - image;
  - target mask;
  - target SDF.
- [ ] Use interpolation appropriate to each representation: bilinear for images
  and SDFs, and nearest-neighbour for discrete masks.
- [ ] Transform a cached SDF as a scalar image by default. Provide an explicit
  option to recompute the SDF from the transformed mask when greater discrete
  accuracy is needed.
- [ ] Store deformed datasets without speckle and add speckle cheaply at sample
  construction time before training. Make its parameters configurable and keep
  validation generation deterministic.

### Deformation-generation calibration

- [ ] Create a visual sweep of the artificial generator parameters:
  - magnitude;
  - integrations;
  - spatial scale(s).
- [ ] Use the sweep to identify useful named generation presets ranging from
  small/local/aggressive to wide/global deformations, with shapes visually
  comparable to deforming a circle into a CCA artery boundary.
- [ ] Keep VoxelMorph architecture and internal deformation tuning fixed during
  this study; revisit them only if later results show a clear need.

### Configurable plaque generation

- [ ] Implement a configuration-driven, parametric plaque generator supporting:
  - plaque count, including more than two plaques;
  - angular positions and separation;
  - plaque sizes;
  - reproducible sampling from configured ranges.
- [ ] Define and validate overlap, boundary, and anatomical-validity rules.
- [ ] Save sampled plaque parameters so individual failures can be inspected.

### Distractors and occlusions

- [ ] Develop image-only fake plaques/blobs that are absent from the ground-truth
  mask and resemble real plaques closely enough to confuse the UNet baseline.
  - Start from structured low-frequency wall echogenicity, but tune shape,
    intensity, position, and frequency content empirically.
  - Use a pilot difficulty sweep rather than assuming low-frequency noise alone
    is sufficiently confusing.
- [ ] Add configurable black rectangles as a controlled ultrasound
  suppression/occlusion artifact.

### Main experiment matrix

- [ ] Generate nine datasets with approximately 2,000 training and 100
  validation samples each:
  1. clean/reference dataset;
  2. same plaque size, easy angle mismatch (for example 10--70 degrees);
  3. same plaque size, hard angle mismatch (for example 70--130 degrees);
  4. different plaque sizes, same angle;
  5. different sizes and angles, easier setting;
  6. different sizes and angles, harder setting;
  7. fake plaques/blobs;
  8. black-rectangle occlusion;
  9. different sizes and angles plus fake plaques.
- [ ] Calibrate the easy/hard ranges before freezing the datasets. Run the hard
  angle setting after inspecting the easy-setting results.
- [ ] Compare the same six UNet-based approaches on all nine datasets:
  1. segmentation-only UNet baseline;
  2. segmentation One-Side SDF + registration One-Side SDF;
  3. segmentation BCE + registration One-Side SDF;
  4. segmentation BCE + registration DSDF MSE;
  5. segmentation BCE + registration One-Side SDF Squared;
  6. segmentation BCE + registration Centroid.
- [ ] Use one seed for the full `9 x 6` matrix; repeat only the most promising
  comparisons with additional seeds.
- [ ] Keep the interpretation tied to the intended stress factor:
  - angle mismatch measures how much angular variation registration can absorb;
  - size mismatch measures how much size variation it can absorb;
  - occlusion is a controlled failure/stress test rather than a claim of full
    clinical realism;
  - fake plaques are the main test of whether the anatomical approach resists
    plausible image-only distractors better than the segmentation-only UNet.

## 3. Multiple-template deformation

- [ ] Start with parallel, independent evaluation of each candidate template.
  Analyze whether candidates should be vectorized across a batch or evaluated
  sequentially based on memory and runtime.
- [ ] After the hard-dataset ablations establish which size and angle changes a
  single template can and cannot compensate for, construct three initial
  templates and later consider four to six.
- [ ] Generate targets with known template provenance and verify whether the
  ideal source template receives the best registration score.
- [ ] Report template-selection accuracy together with segmentation and
  registration quality; also inspect cases where a non-source template produces
  a better fit.

## 4. External datasets

- [ ] Connect the CCA dataset to the current training and evaluation pipeline
  and train without augmentation or domain-specific tuning to establish initial
  real-data numbers.
- [ ] If the CCA pipeline is successful, connect the CSV dataset and determine
  whether a convenient template can be deduced.
- [ ] Connect the ACDC dataset and establish the same untuned baseline pipeline.
- [ ] Defer augmentation, preprocessing sweeps, domain adaptation, and extensive
  tuning until the basic pipelines and metrics work end to end.

## 5. Branch architecture

To be specified later.

## 6. Anatomically constrained network and TEDS-Net

To be specified later.
