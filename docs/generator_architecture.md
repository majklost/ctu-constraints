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
                            compose → deform → rigid
                                      │
                                      ▼
                              image + target labels
```

## Module boundaries

### Configuration and pure geometry

- `generators/types.py` contains validated, serializable configuration and
  parameter dataclasses. It does not perform storage or orchestration.
- `generators/parametrization/` samples plaque parameters and rasterizes empty
  arteries and plaques.
- `generators/composition.py` overlays independent plaque masks onto the empty
  artery. Fake plaques resolve to boundary or lumen targets; real plaques win
  over fake plaques at overlaps.
- `generators/rendering.py` maps target class IDs to grayscale intensities.
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
- `datatools/datasets/composed_artificial_dataset.py` is a read-only consumer.
  For an index it loads selected plaque masks, applies the selected deformation,
  composes targets and grayscale image, then applies the selected rigid preset.
- `scripts/create_*.py` only parse command-line arguments and call the facade.
- `generators/deprecated_generators.py` and
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

The planned SDF cache belongs after composition and deformation but before
rigid transformation. Consequently, its identity depends on the source,
selected plaques and fake-plaque targets, deformation, class-channel order, and
SDF parameters. It does not depend on rigid motion, grayscale intensity, or
image-only noise.

Generation and cache preparation happen outside `Dataset.__getitem__`.
`__getitem__` stays deterministic, read-only, and free of rejection sampling or
cache writes.
