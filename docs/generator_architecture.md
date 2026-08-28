# Generator architecture

The artificial-data code is a composable pipeline. Expensive random geometry
is stored once, while cheap choices remain selectable by the dataset.

```text
source root
   │
   ├── empty artery
   ├── plaque collections ───────────────┐
   └── deformation collections           │
          └── dependent rigid presets    │
                                         ▼
                         ComposedArtificialDataset
                            deform → compose → rigid
                                      │
                                      ▼
                              image + target labels
```

## Module boundaries

### Configuration and pure geometry

- `generators/types.py` contains shared configuration, parameter, and runtime
  layer dataclasses. `SavedPlaque` gives one stored Boolean mask collection its
  target and appearance meanings. `PowerPlaqueSamplingRanges.sample()` resolves
  any number of independent parameter sets from one range configuration.
- `generators/recipes.py` defines the immutable `Recipe`: source, ordered
  plaques, deformation, rigid preset, rendering/noise, and optional SDF cache.
  Named artifacts may carry typed generation backups, allowing the same strict,
  versioned JSON to validate or recreate them on another machine. The complete
  notebook-to-cluster flow is documented in [recipe_workflow.md](recipe_workflow.md).
- `generators/sdf_cache.py` defines the versioned identity, configuration,
  digest, and directory contract for pre-rigid SDF caches. Cache generation
  dispatches through `signed_distance_scipy` or `signed_distance_kornia`
  according to the shared dataset `SDFMode`.
- `generators/parametrization/` rasterizes self-contained empty-artery configs
  and converts tuples of plaque parameters into Boolean union masks.
- `generators/composition.py` overlays independent Boolean plaque masks onto
  target and appearance maps in the exact order supplied. Later layers win at
  overlaps, so precedence is explicit at each call site. A layer's appearance
  defaults to its target class but may be overridden for plaque-like artifacts.
- `generators/rendering.py` maps appearance IDs to grayscale intensities.
- `generators/validation.py` checks topology, foreground margins, and transform
  acceptance.

### Artifact producers

- `generators/source.py` creates the source root and independent named plaque
  collections. It also exposes single-sample plaque generation for tuning.
- `generators/deformation.py` samples, validates, applies, stores, and loads
  backward displacement fields. Collections have shape `[N, 2, H, W]` in
  `(dy, dx)` order.
- `generators/rigid.py` samples and validates `(angle, dx, dy)` presets. A rigid
  preset is nested under its deformation when it depends on one; source-level
  rigid presets represent rigid-only geometry.
- `generators/storage.py` owns small atomic JSON and mmap `.npy` primitives.

Artifact producers depend only on their parent level. Plaque collections are
siblings and independent. A deformation depends on the source artery, while a
deformation-dependent rigid preset depends on that deformation. Deleting a
whole child subtree therefore cannot invalidate a sibling.

### Orchestration and consumption

- `generators/factories.py` is the small script-facing facade. It resolves a
  source root and delegates to the relevant producer; generation algorithms do
  not belong here.
- `generators/recipe_preview.py` resolves backup-only artifacts in memory for
  fast notebook iteration. `generators/recipe_ensure.py` performs a complete
  read-only preflight before it creates, reuses, or explicitly replaces any
  stored artifact.
- `datatools/datasets/composed_artificial_dataset.py` is a read-only consumer.
  It can be constructed from a `Recipe`. For an index it loads the selected
  plaque masks, applies the selected deformation, composes targets and
  grayscale image, then applies the selected rigid preset.
- `scripts/create_*.py` only parse command-line arguments and call the facade.
- `scripts/ensure_recipe.py` is the portable local/cluster entry point for a
  checked-in recipe and its artifact backups.
- `scripts/prepare_composed_artificial_dataset.py` prepares the first large
  source configuration end-to-end, including recipe JSON, deterministic split
  CSVs, and an optional content-addressed SDF cache. It can reuse completed
  artifact boundaries when rerun with the same preparation definition.
- `generators/deprecated/` and
  `scripts/create_artificial_dataset.py` belong to the old fully materialized
  path. New composable behavior should not be added there.

## Device policy

`constraints/devices.py` owns offline compute-device selection. The default
`"auto"` policy is:

```text
CUDA → MPS → CPU
```

Deformation sampling and non-differentiable Kornia SDF computation use this
policy. Explicit `"cuda"` or `"mps"` requests fail early when unavailable.
The selected deformation device is provenance stored in its `config.json`; it
does not change the field layout or application contract.

The differentiable Kornia SDF function used inside losses is excluded from
automatic placement. It must remain on the caller's tensor device so gradients
and model placement are preserved.

## Runtime order and cache boundary

The current composed dataset executes:

```text
load empty artery and selected masks
    → apply non-rigid deformation to each layer
    → compose target labels
    → render grayscale image
    → apply rigid transformation
```

The SDF cache belongs after composition and deformation but before rigid
transformation. `SDFCacheIdentity` is an explicit projection rather than
a hash of the complete recipe: it includes the source ID, ordered plaque names
and target classes, deformation, composition/application contract versions,
class-channel order, and SDF parameters. It excludes rigid motion, plaque
appearance, and grayscale intensity. Adding an unrelated recipe field therefore
does not invalidate the cache; changing an SDF-relevant contract is deliberate
and versioned.

The cache directory is
`derived/sdf-v<identity-version>-<sha256>/`. The digest is calculated from
canonical JSON, while its manifest retains the full identity payload so a
directory is always explainable without reversing its name.

Generation and cache preparation happen outside `Dataset.__getitem__`.
`__getitem__` stays deterministic, read-only, and free of rejection sampling or
cache writes.
