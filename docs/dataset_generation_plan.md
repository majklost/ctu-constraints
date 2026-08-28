# Artificial dataset generation pipeline

## 1. Purpose

Build a deterministic and composable pipeline for artificial artery datasets.
The pipeline must support the harder-dataset experiments in
`docs/upcoming_features.md` without regenerating expensive random deformation
fields for every combination of plaques, rigid motion, image intensities, and
noise.

The system has two output modes:

1. **Lazy composed dataset**: load raw anatomical layers and stored deformation
   fields, compose and deform a sample in `Dataset.__getitem__`, then apply a
   selected rigid preset and image noise.
2. **Materialized dataset**: precompute one frequently used lazy recipe into
   ordinary image, label, and optional SDF arrays. This is the fallback when a
   lazy recipe cannot feed training quickly enough and is also the format most
   similar to future real-world datasets.

Dataset loading must be read-only. A loader may consume an existing derived
cache, but it must never create or modify a cache from `__getitem__`.

## 2. Terminology and semantic contracts

Use the following terms consistently in code, manifests, and documentation.

- **Registration template**: one source segmentation sampled or configured once
  for the whole dataset. It is passed to the registration model and is not the
  per-sample raw anatomy.
- **Raw sample**: one independently sampled artery and its independently sampled
  real and fake plaques before random deformation. Fixing sampling ranges to a
  single value is how an experiment requests identical raw geometry across
  samples.
- **Artery labels**: the base label map containing only background, boundary,
  and lumen.
- **Real plaque mask**: the Boolean union of all real plaques in one sample.
- **Fake plaque mask**: the Boolean union of all fake plaques in one sample.
  Fake plaques affect image synthesis but are never assigned the real-plaque
  target class.
- **Synthesis labels**: the composed class-ID map used to create the grayscale
  image. It may contain `ArteryClass.FAKE_PLAQUE`.
- **Target labels**: the anatomical class-ID map used for training and SDF
  calculation. It contains no `FAKE_PLAQUE`; fake-plaque pixels are mapped to a
  configured non-plaque class, initially boundary or lumen.
- **Deformation preset**: one stored random displacement field per sample plus
  the configuration and convention used to produce those fields.
- **Rigid preset**: one stored `(angle, dx, dy)` triple per sample, validated
  against the samples produced by one deformation preset.
- **Noise preset**: deterministic runtime noise configuration or, for expensive
  noise only, a stored per-sample noise array.
- **Dataset recipe**: the complete immutable selection of source layers,
  plaques, deformation, rigid transformation, target mapping, grayscale
  intensities, SDF policy, and noise.

The initial synthesis IDs are defined by `ArteryClass`:

```text
0 background
1 boundary
2 lumen
3 real plaque
4 fake plaque (synthesis only)
```

The training target schema remains four-class:

```text
0 background
1 boundary
2 lumen
3 real plaque
```

All stored label maps are class-ID maps, not one-hot masks. The loader converts
them to tensors required by the existing `Sample` and `LabelSchema` contracts.

## 3. Determinism

Every random result must be a pure function of the dataset seed, sample index,
operation stream, and rejection attempt:

```python
np.random.SeedSequence([dataset_seed, sample_index, stream_id, attempt])
```

Reserve stable stream IDs in one module. At minimum define separate streams for:

```text
0 artery parameters
1 real-plaque parameters
2 fake-plaque parameters
3 deformation field
4 rigid parameters
5 speckle
6 structured noise/distractors
7 black-rectangle occlusion
```

Adding or disabling one operation must not change another operation's random
values. Seeds must not depend on DataLoader worker ID, worker count, batch size,
access order, or global NumPy state.

Torch operations that do not accept a `torch.Generator` must be isolated from
the caller's global RNG state. Generation scripts may use `torch.random.fork_rng`
and seed it with the per-sample stream seed. Store the master seed, seed scheme
version, dependency versions, and Git commit in the manifest.

Seeds and configuration provide provenance but do not promise byte-identical
regeneration across arbitrary future library versions or hardware. Stored
arrays are the authoritative dataset outputs.

## 4. On-disk layout

A composable synthetic source dataset has this logical layout:

```text
<dataset>/
    manifest.json
    completed.npy

    template/
        target_labels.npy
        parameters.json

    raw/
        artery_labels.npy
        real_plaque_masks.npy
        fake_plaque_masks.npy
        parameters.jsonl

    deformations/
        <preset>/
            manifest.json
            fields.npy
            completed.npy
            diagnostics.jsonl
            rigid/
                <preset>.npy
                <preset>.json
                <preset>.diagnostics.jsonl

    noise/
        <preset>.json
        <expensive-preset>/
            manifest.json
            values.npy

    derived/
        <recipe-hash>/
            manifest.json
            sdf_scipy.npy
            completed.npy
```

Raw arrays use these contracts:

| File | Shape | dtype | Meaning |
|---|---:|---|---|
| `template/target_labels.npy` | `[H, W]` | `uint8` | Dataset registration template |
| `raw/artery_labels.npy` | `[N, H, W]` | `uint8` | IDs 0, 1, and 2 only |
| `raw/real_plaque_masks.npy` | `[N, H, W]` | `bool` | Union of all real plaques |
| `raw/fake_plaque_masks.npy` | `[N, H, W]` | `bool` | Union of all fake plaques |
| `completed.npy` | `[N]` | `bool` | Successfully written raw samples |

Version 1 selects real and fake plaques at category level (`include` true or
false). Individual plaque parameters remain separate in JSONL, but overlapping
same-category plaques are stored as one union mask. Supporting selective
per-instance composition later requires a new format version with separate
per-instance mask channels; it must not reinterpret these union masks.

Deformation arrays use:

| File | Shape | dtype | Meaning |
|---|---:|---|---|
| `fields.npy` | `[N, 2, H, W]` | `float32` | Backward sampling displacement in `(dy, dx)` spatial-axis channel order |
| `completed.npy` | `[N]` | `bool` | Successfully written or explicitly fallen-back fields |

Do not change the field dtype without a numerical comparison and anatomical
validation. One float32 preset with 2,000 samples at 256 by 256 pixels is about
1 GiB.

Rigid arrays use:

| File | Shape | dtype | Meaning |
|---|---:|---|---|
| `rigid/<preset>.npy` | `[N, 3]` | `float32` | `(angle_radians, dx_pixels, dy_pixels)` |

The paired rigid JSON file defines the exact convention and rejection bounds.

All bulk arrays must be valid `.npy` files created with
`np.lib.format.open_memmap`, so they can later be opened with
`np.load(..., mmap_mode="r")`. Object-dtype arrays and pickled NumPy objects are
forbidden.

## 5. Manifests and per-sample metadata

### 5.1 Root manifest

`manifest.json` is dataset-wide and must contain at least:

- `format_name` and integer `format_version`;
- immutable `dataset_id`;
- `status`: `incomplete` or `complete`;
- requested sample count and valid sample count;
- image height and width;
- array filenames, shapes, and dtypes;
- class IDs, names, and SDF foreground-channel order;
- dataset seed and seed-scheme version;
- complete artery, real-plaque, and fake-plaque sampling configuration;
- registration-template configuration;
- generator and rasterization configuration;
- creation timestamp;
- Git commit and dirty-worktree status when available;
- Python, NumPy, Torch, SciPy, Kornia, Neurite, and project versions;
- parameter-angle units and all spatial units;
- fallback and rejection policy;
- counts of accepted, rejected, and fallback samples.

The `dataset_id` is an immutable identity assigned at source-dataset creation
and preserved when that dataset is copied or moved. It must not depend on an
absolute filesystem path and may be a UUID. It is not a content hash: derived
manifests reference this ID and additionally store hashes of the configurations
and artifacts they consume.

### 5.2 Per-sample parameters

In a complete dataset, `raw/parameters.jsonl` contains exactly one JSON object
per requested sample in sample-index order. Each record contains:

- `sample_index` and stable `sample_id`;
- resolved saveable artery parameters, including image size, center, lumen
  radius, and wall thickness;
- real plaques as a list of `{type, parameters}` records;
- fake plaques as a list of `{type, parameters}` records;
- stream seeds or the information required to derive them;
- sampling attempt count;
- validation result and any expected fallback information.

For the current implementation, `type` is `"power"` and `parameters` is the
JSON representation of `PowerPlaqueParameters`. Never serialize `_PlaqueSpec`,
because it contains runtime callables.

JSONL is used because plaque count is variable. It must be written by one
coordinator process, not concurrently by workers. During resumable generation,
the coordinator may append records out of order, keyed by unique
`sample_index`. Before marking the dataset complete, it must reject duplicate or
missing indices and atomically rewrite the final JSONL in index order.

### 5.3 Derived recipe identity

Do not hash the complete dataset recipe for a derived cache. Each cache type
defines an explicit, versioned identity projection containing only inputs that
can change its values. The current pre-rigid SDF projection includes the source
`dataset_id`, ordered plaque collection names and target classes, deformation,
composition and deformation-application contract versions, foreground-channel
order, and SDF implementation settings.

It deliberately excludes plaque appearance, class intensities, and rigid
motion. Therefore adding an unrelated field to `Recipe` does not invalidate SDF
caches. Adding a new SDF-relevant input requires deliberately extending and
versioning `SDFCacheIdentity`.

### 5.4 Dataset recipe contract

`Recipe` is a frozen dataclass in code and has a strict JSON representation for
experiment provenance. The current recipe also owns its data-root-relative
source path, optional noise and SDF configuration, and typed generation backups
for named plaques, deformation, and rigid artifacts. This makes the tracked JSON
portable between a tuning notebook and a cluster. Omitting deformation or rigid
motion is represented by `null` and means absence of that transformation.

Unknown fields, versions, class names, and missing intensities for enabled
appearances are initialization errors. The current contract and worked workflow
are documented in [recipe_workflow.md](recipe_workflow.md); resolved recipes
belong in the tracked `recipes/artificial/` directory.

## 6. Raw anatomy generation

### 6.1 Dataset-level registration template

Sample or construct the registration template once. Save its four-class target
label map and its complete serializable configuration. Template SDFs may be
computed by the dataset loader or stored as explicit optional artifacts.

The registration template is separate from all per-sample raw layers.

### 6.2 Per-sample artery

For each sample index:

1. Derive the artery RNG stream.
2. Sample a saveable artery parameter object.
3. Render an artery label map containing only background, boundary, and lumen.
4. Validate shape, dtype, allowed class IDs, required classes, connectivity,
   wall integrity, and configured border margin.

Separate `ArterySpec` into a serializable parameter representation and the
resolved runtime `ArterySpec` when necessary. Runtime plaque callables must not
appear in the serializable representation.

### 6.3 Real and fake plaque masks

Plaque samplers receive the resolved artery configuration so their sampled
parameters are compatible with lumen radius and wall thickness. For each
plaque:

1. Sample serializable parameters using the plaque-specific RNG stream.
2. Construct the runtime plaque with the registered plaque factory.
3. Rasterize it as an individual temporary Boolean mask without permanently
   overwriting the artery labels.
4. Validate the individual plaque and merge it into the corresponding stored
   real or fake union mask.

Initial validity rules are:

- positive inward protrusion;
- configured minimum wall embedding;
- protected outer-wall margin;
- minimum plaque pixel area;
- configured minimum angular separation when overlap is forbidden;
- configured maximum overlap when overlap is allowed;
- maximum combined angular coverage and lumen obstruction;
- every significant real plaque touches both lumen and boundary;
- fake plaques obey the same geometric placement rules unless their
  configuration explicitly defines a different rule.

Real plaques take priority over fake plaques where they overlap. Individual
parameters and validation results remain separate in JSONL even though the
stored masks are same-category unions.

### 6.4 Composition

Composition starts from the artery labels and overlays selected plaques using
this default priority:

```text
artery < fake plaque < real plaque
```

Produce `raw_synthesis_labels` by assigning `FAKE_PLAQUE` to selected fake-mask
pixels and `PLAQUE` to selected real-mask pixels. Produce `raw_target_labels`
from the synthesis map by replacing every fake-plaque pixel with the recipe's
configured `fake_plaque_target`, initially `BOUNDARY` or `LUMEN`.

Validate the composed raw target with
`does_violation_occur_with_wall` and the additional configured area and border
checks. The composition functions must be pure and separately unit tested.

## 7. Deformation presets

### 7.1 Field generation

Use `constraints.voxelmorph.utils.random_disp`. A deformation preset manifest
must specify:

- source `dataset_id`, `N`, `H`, and `W`;
- `scales`;
- `magnitude`;
- `integrations`;
- `voxsize`;
- `fractal_mode`, with `blur` preferred for final datasets;
- field shape, dtype, channel order, units, and transform direction;
- seed scheme and stream ID;
- interpolation and discretization conventions used for validation;
- maximum attempts and fallback policy.

Generate one field for every source sample. Field generation is deterministic
per index and may be parallelized, but only the coordinator process writes the
shared memmap.

The initial field convention is exactly the convention consumed by
`constraints.voxelmorph.utils.spatial_transform(..., isdisp=True)`: channels
follow spatial-axis order `(dy, dx)`, values are measured in pixels, and an
output location samples the input at `location + displacement(location)`. This
is a backward sampling field; it must not be described as forward content
movement. Verify the convention with landmark tests rather than relying only on
channel names.

### 7.2 Application and discretization

At loading or materialization time:

1. Compose the selected raw synthesis labels.
2. Convert the synthesis labels to one-hot channels.
3. Apply the stored field to all channels in one `spatial_transform` call using
   linear interpolation.
4. Convert the soft result back to class IDs with `argmax`.
5. Derive deformed target labels by applying the configured fake-plaque target
   mapping.

The same operation without fake plaques must expose the underlying artery or
real plaque rather than a label baked under the fake plaque. This is why raw
artery and plaque masks are stored separately.

### 7.3 Post-deformation validation

Field preparation must validate the field against the default full raw
composition for its source sample. Validation includes:

- allowed class IDs and presence of required classes;
- configured minimum foreground border margin;
- no unacceptable per-class area loss;
- anatomical constraint validation;
- optional Jacobian determinant thresholds for folding diagnostics.

Use cheap array checks before connected-component constraint checks. Retry with
the next deterministic attempt seed up to `max_attempts`.

If all expected deformation attempts fail, use the zero/identity field so the
valid raw sample is preserved. Record the fallback in the preset manifest and a
per-sample record in `diagnostics.jsonl`. Do not silently substitute it.

Because optional plaque-category inclusion can change anatomical validity, recipe
preparation or materialization must validate every final recipe once. Full
anatomical validation is not performed on every training read.

## 8. Rigid presets

Rigid displacement is applied after non-rigid deformation. Each preset stores
one triple per sample:

```text
(angle_radians, dx_pixels, dy_pixels)
```

Use these geometric conventions:

- coordinates are image coordinates;
- `dx > 0` moves content to the right;
- `dy > 0` moves content downward;
- positive angle means counter-clockwise movement of image content;
- rotation is about the image center;
- stored parameters describe the forward movement of content, not PyTorch's
  backward sampling grid;
- conversion to `affine_grid` coordinates is centralized in one tested helper;
- the JSON records the chosen `align_corners` value.

Do not pass stored pixel translations directly to the existing
`differentiable_rigid`, which expects normalized grid translations. Add or use
an explicit conversion helper and test it with single-pixel landmarks.

For each sample:

1. Load and apply its deformation to the base artery foreground.
2. Sample a rigid triple from the preset ranges.
3. Transform the discrete foreground with nearest-neighbour interpolation.
4. Reject if the foreground touches the configured margin, loses unacceptable
   area, loses required classes, or violates configured anatomical predicates.
5. Retry up to `max_attempts` using deterministic attempt seeds.
6. Store identity `(0, 0, 0)` after exhaustion and record the fallback.

Write per-sample attempts, rejection reasons, accepted parameters, minimum
border margin, area change, and fallback status to
`<preset>.diagnostics.jsonl`.

Plaques are constrained inside the outer artery, so rigid acceptance should be
based primarily on the deformed artery extent. This permits one rigid preset to
be reused across internal plaque configurations. If a future generator allows an
artifact outside the artery contour, the rigid preset must become dependent on
that composition.

At loading time apply an identical stored rigid transform to:

- image: bilinear interpolation;
- target labels: nearest-neighbour interpolation;
- cached pre-rigid SDF: bilinear interpolation.

Support an explicit strict mode that recomputes the SDF from the transformed
discrete target rather than transforming the cached SDF.

## 9. Grayscale synthesis and image-only noise

Create the grayscale image after deformation and before rigid movement by
mapping synthesis class IDs to configured finite float intensities. The default
output contract is `[1, H, W]`, `float32`.

Because class-to-intensity mapping is cheap, do it at runtime. This permits the
same geometry to be evaluated with different intensity mappings. If future
texture synthesis becomes expensive, introduce a separate derived image cache
instead of complicating the label cache.

Apply image-only effects after rigid movement in this order:

```text
structured distractor/noise
    -> black-rectangle occlusion
    -> speckle
```

They must never modify target labels or SDFs.

Generate ordinary Gaussian or multiplicative speckle deterministically at
runtime; storing it wastes substantial disk space. A noise preset uses stored
arrays only when generating that noise is demonstrably expensive. Validation
noise uses fixed per-sample streams. A separately named `per_epoch` mode may be
added later, but it is not used for the controlled experiment matrix.

## 10. SDF policy and derived caches

SDFs are computed from the final discrete target labels, excluding the
background channel. Their shape is `[C_fg, H, W]`, dtype is `float32`, channel
order follows `LabelSchema.foreground_ids`, and the sign convention is negative
inside and positive outside.

The loader supports these explicit policies:

- `disabled`: do not return an SDF;
- `require_cache`: fail during dataset initialization if the exact recipe cache
  is absent or incomplete;
- `compute`: calculate the SDF in memory on each access and never write it;
- `transform_cached`: load a pre-rigid cached SDF and apply the selected rigid
  transform bilinearly;
- `strict_after_rigid`: recompute from rigidly transformed target labels. This
  may use a materialized cache whose key includes the rigid preset.

SciPy is the default SDF implementation for CPU loading. A local 256 by 256,
three-channel benchmark measured approximately 12 ms per sample for SciPy and
424 ms for Kornia on CPU. Treat these as calibration values, not permanent
performance guarantees. Do not use uncached Kornia SDF generation in ordinary
CPU DataLoader workers.

Derived cache creation is an explicit script operation. It uses a temporary or
incomplete state, a completion bitmap, and an atomic final manifest update.
Readers reject incomplete or recipe-mismatched caches.

## 11. Lazy dataset reader

Implement a new composable artificial dataset rather than accumulating optional
branches in the legacy `CachedArtificialDataset`. Its constructor receives a
validated immutable `DatasetRecipe` and resolves all paths and manifests before
the first call to `__getitem__`.

It must return the existing `Sample` contract:

```text
image           [1, H, W], float tensor
target_labels   [H, W], long tensor
sample_id       stable string
sdf             [C_fg, H, W], float tensor when requested
template        [H, W], long tensor when requested by the current pipeline
```

Optional diagnostic transforms or fields must use clearly named keys and must
not overload the legacy `transform.npy` meaning.

`__getitem__` must:

- perform no writes;
- use no global RNG state;
- perform no full anatomical validation in normal training mode;
- copy read-only NumPy memmap slices before passing them to Torch when mutation
  is possible;
- apply the stored deformation to all synthesis channels in one operation;
- verify final tensor shapes and target class IDs;
- produce the same result regardless of worker count and access order.

For training, start with one DataLoader using approximately:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
)
```

Set `torch.set_num_threads(1)` in worker initialization to avoid every worker
creating a full CPU thread pool. DataLoader workers must not initialize or use
CUDA. Tune worker count by measuring GPU idle time and batch latency on the
actual storage and training machine.

## 12. Materialized datasets

Provide a `datasetcacher` or `materialize_artificial_dataset.py` script that
consumes one complete recipe and writes an immutable fully cached dataset:

```text
<materialized-dataset>/
    manifest.json
    completed.npy
    images.npy
    target_labels.npy
    sdf_scipy.npy          # optional
    template_labels.npy
```

Contracts are:

| File | Shape | dtype |
|---|---:|---|
| `images.npy` | `[N, 1, H, W]` | `float32` |
| `target_labels.npy` | `[N, H, W]` | `uint8` |
| `sdf_<mode>.npy` | `[N, C_fg, H, W]` | `float32` |
| `template_labels.npy` | `[H, W]` | `uint8` |
| `completed.npy` | `[N]` | `bool` |

The materialized manifest embeds the full source recipe and parent artifact
identities. A generic materialized reader should use the same external `Sample`
contract so future real-world dataset adapters can converge on the same tensor
shapes.

Noise belongs in a materialized image only when the recipe specifies fixed
noise. Per-epoch augmentation is not materializable.

## 13. Parallel generation, resume, and failures

Use worker processes for pure indexed computation and one coordinator process
for shared-file writes. Workers return `(sample_index, outputs, metadata)`.
Workers must never write to a shared `.npy`, JSONL, manifest, or completion
bitmap.

Preallocate memmaps for the requested sample count. After the coordinator has
written and flushed every output for an index, it sets `completed[index] = True`.
On restart, validate the existing manifest and arrays, then submit only
incomplete indices. Deterministic index seeds guarantee that resumed output
matches uninterrupted generation.

Catch only expected, typed generation failures such as invalid sampled anatomy
or exhausted rejection attempts. Unexpected exceptions, shape mismatches,
serialization errors, and disk errors stop generation and leave the manifest
as `incomplete`.

Expected rejection exhaustion uses the explicitly configured identity/raw
fallback and records it, preserving exactly `N` usable indices. If a future
command instead permits skipped samples, it must expose a valid-index mapping
and accurate valid count; unwritten memmap entries must never appear as valid
samples.

Mark `status` as `complete` only after:

- all required completion bits are true;
- all arrays have the declared shape and dtype;
- all files are flushed;
- parameter-record count equals requested sample count;
- validation/fallback summaries are written.

## 14. Central storage API

Keep disk-format knowledge centralized, while allowing stages to declare
different outputs. Implement small focused abstractions rather than a saver
that serializes arbitrary Python values:

- manifest dataclasses and validation;
- `ArtifactSpec(name, relative_path, shape, dtype)`;
- single-writer memmap creation/open/resume helper;
- composable source reader;
- derived-cache reader/writer used only by preparation scripts;
- materialized dataset reader/writer;
- recipe normalization and hashing;
- explicit serializers for artery and plaque parameter records.

Every reader validates format version, artifact existence, shape, dtype, source
dataset identity, sample count, and completion status before training begins.

## 15. Backward compatibility

Do not force the new format to reuse ambiguous legacy filenames. In particular,
the old `transform.npy` may contain either a rigid matrix or a deformation field.

Keep the current `CachedArtificialDataset` working for existing datasets during
migration. Add a separate new reader for this format. If desired later, provide
an explicit migration or adapter command that maps:

```text
img.npy       -> images.npy
channel mask  -> target_labels.npy
template.npy  -> template_labels.npy
```

Legacy rigid matrices require an explicit documented conversion and must not be
silently interpreted as new `(angle, dx, dy)` arrays.

## 16. Proposed module and script responsibilities

The exact class names may change, but responsibilities should remain separated:

```text
constraints/generators/types.py
    serializable artery configuration, label/array aliases, shared contracts

constraints/generators/parametrization/
    plaque parameter dataclasses, samplers, factories, rasterization

constraints/generators/composition.py
    pure artery/plaque composition and fake-target mapping

constraints/generators/deformation.py
    deformation configuration, deterministic field generation, application,
    discretization, and validation

constraints/generators/rigid.py
    rigid bounds, pixel-coordinate convention, sampling, conversion,
    application, and rejection

constraints/generators/noise.py
    deterministic image-only noise callables and configurations

constraints/generators/storage.py
    manifests, artifact specs, memmaps, resume, and JSONL serialization

constraints/generators/recipes.py
    immutable dataset recipes, validation, canonical JSON, and hashes

constraints/datatools/datasets/composed_artificial_dataset.py
    read-only lazy Dataset implementation

constraints/datatools/datasets/materialized_artificial_dataset.py
    simple fully cached Dataset implementation

scripts/create_artificial_source.py
    registration template and raw layer generation

scripts/create_deformation_preset.py
    field generation and validation

scripts/create_rigid_preset.py
    rigid generation and rejection

scripts/cache_artificial_sdf.py
    explicit derived SDF cache generation

scripts/materialize_artificial_dataset.py
    complete recipe materialization
```

Do not add new behavior to `generators/deprecated/`. Use it only as a reference
until compatibility is no longer needed.

## 17. Implementation sequence

Implement and review the work in this order:

1. Define manifest, artifact, serializable parameter, and recipe contracts.
2. Implement deterministic seed streams and unit tests.
3. Separate serializable artery parameters from runtime `ArterySpec`.
4. Generate and store artery, real-plaque, and fake-plaque layers.
5. Implement pure composition and anatomical validation.
6. Implement single-writer memmap storage, completion tracking, and resume.
7. Implement deformation preset generation, application, and fallback.
8. Implement rigid conventions, conversion, rejection, and preset storage.
9. Implement the lazy composed dataset without SDF caching.
10. Add SciPy SDF calculation and explicit derived-cache support.
11. Add deterministic runtime speckle and black-rectangle occlusion.
12. Implement the materializer and materialized reader.
13. Benchmark DataLoader throughput and tune workers.
14. Produce visual deformation and artifact calibration sweeps.

Do not implement every future distractor before the source, deformation, rigid,
and materialization paths work end to end.

## 18. Required automated tests

### Determinism

- Identical seed, index, stream, and attempt produce identical parameters and
  arrays.
- Results are identical for zero workers and multiple workers.
- Enabling fake plaques or noise does not change artery, deformation, rigid, or
  unrelated noise streams.
- Resumed generation matches uninterrupted generation.

### Storage

- Every `.npy` opens with `np.load(..., mmap_mode="r")` and has declared shape
  and dtype.
- Readers reject incomplete datasets, wrong sample counts, wrong parent IDs,
  mismatched shapes, and unsupported format versions.
- Variable plaque counts round-trip through JSONL without pickle.
- A recipe hash is stable under dictionary key ordering and changes when a
  target-affecting option changes.

### Anatomy and composition

- Artery layers contain only background, boundary, and lumen IDs.
- Stored real and fake masks equal the union of their serialized plaques.
- Real plaques override fake plaques on overlap.
- Disabling a fake plaque reveals the underlying artery or real plaque.
- Mapping fake plaques to boundary or lumen never leaves synthesis-only class 4
  in target labels.
- Generated raw and representative composed targets satisfy anatomical rules.

### Deformation

- Zero field is an identity transform.
- Linear one-hot warping followed by `argmax` produces only valid class IDs.
- The identical field is applied to every synthesis channel.
- Accepted fields satisfy border, area, and anatomical predicates.
- Exhausted attempts produce a recorded identity fallback.
- Stored field channel order and direction are verified with a synthetic
  landmark translation.

### Rigid transformation

- Zero parameters are identity.
- Positive `dx`, positive `dy`, and positive angle move landmarks according to
  the documented forward convention.
- Image, target, and SDF receive the same geometric transform with their
  configured interpolation methods.
- Nearest-neighbour target transformation introduces no invalid label IDs.
- Accepted transforms preserve border margin and configured area tolerance.
- Exhausted attempts produce recorded identity parameters.

### SDF and image artifacts

- Cached and freshly computed SDFs match within implementation-specific
  tolerance.
- SDF channel order and sign convention are correct.
- Bilinearly transformed cached SDF is compared with strict recomputation after
  rigid motion and the discrepancy is reported in a calibration test.
- Grayscale intensity mapping is correct for every synthesis class.
- Noise and occlusion never modify target labels or SDFs.
- Fixed noise is deterministic per sample.

### End-to-end

- A small lazy recipe returns the existing `Sample` contract and collates in a
  DataLoader.
- Materializing that recipe preserves image, target, template, and SDF values.
- A training smoke test can consume both lazy and materialized readers.
- A throughput benchmark reports samples per second, batch wait time, and GPU
  utilization for selected worker counts.

## 19. Calibration before the main experiment matrix

Before freezing the nine datasets from `docs/upcoming_features.md`:

1. Generate visual sweeps over deformation magnitude, integrations, and spatial
   scales for both local and global presets.
2. Record raw and post-deformation rejection/fallback rates.
3. Inspect worst border margins, class-area changes, and Jacobian diagnostics.
4. Compare SciPy and Kornia SDF output and runtime.
5. Compare transformed cached SDF with strict post-rigid recomputation.
6. Measure lazy DataLoader throughput with 0, 1, 2, 4, and 8 workers on the
   actual training machine.
7. Materialize recipes whose input pipeline leaves the GPU waiting materially
   or whose exact configuration will be reused many times.
8. Calibrate easy and hard rigid/deformation presets before generating the full
   experiment matrix.

The experiment manifests must retain the complete recipe so every training run
can be traced back to its raw source, deformation preset, rigid preset, SDF
policy, intensity mapping, and noise configuration.
