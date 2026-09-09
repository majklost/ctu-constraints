# TEDS-Net integration

```
@article{wyburd_teds-net_2024,
 title = {{TEDS}-{Net} {UPDATED}},
 volume = {97},
 issn = {13618415},
 shorttitle = {Anatomically plausible segmentations},
 url = {https://linkinghub.elsevier.com/retrieve/pii/S1361841524001476},
 doi = {10.1016/j.media.2024.103222},
 language = {en},
 urldate = {2026-08-23},
 journal = {Medical Image Analysis},
 author = {Wyburd, Madeleine K. and Dinsdale, Nicola K. and Jenkinson, Mark and Namburete, Ana I.L.},
 month = oct,
 year = {2024},
 pages = {103222},
}
```

Vendored, unmodified upstream code lives in [constraints/teds_net/](../constraints/teds_net/)
(1503 lines total). Upstream: <https://github.com/mwyburd/TEDS-Net>.

## Goals

- **Track A — comparison.** A faithful TEDS-Net, wrapped in its own Lightning module,
  reusing this repo's metric/loss/template/logging modules, so its numbers land in the
  same W&B panels as everything else.
- **Track B — extraction.** Pull the folding-prevention machinery
  (tanh half-grid activation, smoothed scaling-and-squaring, super-resolution field)
  out into components that accept an N-channel input (e.g. image ⊕ mask concatenation)
  and emit a deformation field — i.e. drop-in swappable with `VxmPairwise`.

---

## 1. Architecture: decisions already made

| # | Decision | Choice |
|---|---|---|
| D1 | How TEDS-Net enters the framework | Faithful standalone module + its own `TedsLightning`, reusing existing metrics/losses/template sources. Track B extracts components later. |
| D2 | Dataset ladder | (a) MNIST on vendored code, smallest possible change → (b) MNIST through this repo's dataset harness → (c) ACDC / artificial, which should then be nearly free. |
| D3 | Field units | TEDS-Net internals stay in normalized `[-1,1]` grid units; convert to **pixel units at the boundary** where the field is handed to metrics/losses. |

### D3 in detail — the unit conversion

This is the single most important interface detail, and it is silent when you get it wrong.

TEDS-Net's transformer builds its grid as `torch.linspace(-1, 1, s)` and does
`new_locs = grid + flow` with `align_corners=True`
([utils_teds.py:172-206](../constraints/teds_net/network/utils_teds.py#L172-L206)).
So its flow is in **normalized** units. With `align_corners=True`, the span `[-1, 1]`
covers pixel centres `0 … s-1`, hence:

```
1 normalized unit  =  (s - 1) / 2  pixels
field_pixels[c]    =  field_norm[c] * (size[c] - 1) / 2
```

Three consequences:

1. **The factor is per-axis, not a scalar.** For ACDC at 144×208 the two channels scale
   by different amounts. A single scalar silently shears the field.
2. **Channel order already matches.** TEDS-Net's `new_locs[..., [1, 0]]`
   ([utils_teds.py:200](../constraints/teds_net/network/utils_teds.py#L200)) means channel 0
   follows the first spatial axis (y/H) — the same convention
   `DeformationJacobianTerm` assumes ("component 0 follows the first spatial (y) axis",
   [metric_terms.py:229-233](../constraints/computers/metric_terms.py#L229-L233)). No flip needed,
   only scaling.
3. **Without the conversion, folding detection reads as clean.** `DeformationJacobianTerm`
   computes `(1 + du_dy)(1 + du_dx) - …`. A normalized-unit field has `|du| ~ 1e-2`, so
   every determinant lands near `1.0` and the metric reports ~0% folding *regardless of
   what the field does*. Same for `DeformationGradientTerm`. This would look like a
   passing result.
4. **Use the mega-resolution size.** The paper measures `% |J_Φ| ≤ 0` on the final
   upsampled field `Φ ∈ R^{2×2H×2W}`. So convert with `size = Mega_inshape = mega_P * inshape`
   and report the Jacobian at that resolution to be comparable to Table 1/2.

### What already exists here that you do *not* need to rebuild

- `DeformationJacobianTerm` ([metric_terms.py:200](../constraints/computers/metric_terms.py#L200)) —
  this *is* the paper's `% |J_Φ| ≤ 0` folding metric, already reporting per-pixel fraction,
  per-sample fraction, and minimum determinant.
- `ACDCSegmentationConstraintViolationTerm` / `ACDCRegistrationConstraintViolationTerm`
  ([metric_terms.py:374-460](../constraints/computers/metric_terms.py#L374-L460)) plus the annularity
  check in [constraint_function.py](../constraints/losses_metrics/constraint_function.py) — a
  Betti-`{b0=1, b1=1}` proxy for the myocardium.
- `minimum_jacobian_determinant` in [generators/validation.py](../constraints/generators/validation.py) —
  a numpy Jacobian already used for deformation rejection.
- `DeformationGradientTerm` ([loss_terms.py:95](../constraints/computers/loss_terms.py#L95)) — the
  field regularizer, though note §3 below: it penalises ∇u, TEDS-Net penalises ∇Φ.
- `PerSampleTemplateSource` / `TemplateBatch` — the prior-shape plumbing.
- `iter_deformation_fields` — walks `TransformSpec.steps`, so a two-branch (bulk + ft)
  spec gets both fields into the metrics for free.

**Gap:** there is no Betti-number computation in this repo (grep for `betti` returns nothing).
For MNIST digits (b1 ∈ {0,1,2}) you need one; see task **P1.3**.

### The one structural friction with Track A

`SegmentationRegistrationModel.forward` ([composed.py](../constraints/models/composed.py)) always
runs `segmentation_net(x)` first and feeds *that* to the registration net. TEDS-Net is
**image-fed**: it has no segmentation branch at all — warping the prior *is* the segmentation.
This is exactly why D1 (own Lightning wrapper) is the right call for Track A: don't bend
`ProjectLightning` around a model that has one output where it expects two.

What `TedsLightning` must therefore do differently from `ProjectLightning`:

- `segmentation_logits = None` in `LossInput`/`MetricInput`. Only registration-side terms apply.
  `SegmentationIoUTerm` will need skipping (build the composite without it) — the warped prior
  goes to `RegistrationIoUTerm` instead.
- No `SpatialTransformer` call in the wrapper. TEDS-Net warps the prior *internally*, at 2×
  resolution, and max-pools the result. Extracting that would break the method.
- Emit the (unit-converted) bulk and ft fields as `TransformSpec(steps=(...))` purely so the
  Jacobian/grad terms can see them — the spec is diagnostic output here, not an instruction
  to warp.

---

## 2. Paper vs. vendored code — audit

Hyperparameters match the paper wherever the release exposes them: ACDC uses `h=8`, `σ=2`,
`k=5`, `α=β=10000`, `T=0.3`, `lr=1e-4`, `batch=5`, 200 epochs, `f=12`, `l=4`,
`l_bulk=4`, `l_ft=2`, prior radius 35 / thickness 7. MNIST uses `α=150`, `σ=2`, `k=3`, `h=8`.
So the *method* is faithfully implemented. What is missing is everything around it.

### 2a. Confirmed bugs

| # | Issue | Where |
|---|---|---|
| B1 | **The MNIST mock example feeds the binary label as the network input.** `X_set` is a numpy *view* into `Y_set`, so the in-place binarisation at line 47-48 mutates the image too. Verified numerically. The README's `Test Dice 0.927` is therefore *not* a segmentation result — the input is the answer. | [mnist.py:33-48](../constraints/teds_net/dataloaders/mnist.py#L33-L48) |
| B2 | **The ACDC dataloader loads the same file for image and label** (`Vol/{ID}` twice). Same class of bug as B1. | [ACDC.py:47-48](../constraints/teds_net/dataloaders/ACDC.py#L47-L48) |
| B3 | `grad_loss` hardcodes `'cuda:0'` → crashes on CPU-only runs. | [losses.py:51-53](../constraints/teds_net/utils/losses.py#L51-L53) |
| B4 | `grad_loss` is 2D-only (indexes dims 2,3) despite carrying `self.ndims`. Silently wrong in 3D. | [losses.py:41](../constraints/teds_net/utils/losses.py#L41) |
| B5 | `GaussianSmoothing` has `kernel_dic = {3:1, 5:2}` → `KeyError` for any other kernel size. | [utils_teds.py:232](../constraints/teds_net/network/utils_teds.py#L232) |
| B6 | The learnable-σ branch (`sigma < 0`, not in the paper) hardcodes 2 channels via `torch.cat((kernel,kernel),dim=1)` — broken in 3D — and uses a dense conv without `groups`, so it is not depthwise as the docstring claims. | [utils_teds.py:257-274](../constraints/teds_net/network/utils_teds.py#L257-L274) |
| B7 | `do_evalutation` sets `self.params.batch=1` *after* the dataloaders were built → no effect. | [trainer.py:125](../constraints/teds_net/trainer.py#L125) |
| B8 | `ConvBlock.__init__` is missing `self`. Harmless only because the class is never instantiated — only `ConvBlock._block` is called. | [UNet.py:20](../constraints/teds_net/network/UNet.py#L20) |
| B9 | `DecoderBranch` always allocates all four up/decode stages regardless of `net_depth`/`dec_depth`, so unused parameters inflate any "# Parameters" comparison against Table 1. | [UNet.py:160-190](../constraints/teds_net/network/UNet.py#L160-L190) |
| B10 | **The ACDC path cannot run at all.** `setup_acdc_dataloader` calls `MyDataset(params, dataset_dict['train'], subset='Train', aug=True)`, but `ACDC_dataclass.__init__` is `(self, params, subset)` — wrong arity *and* an unknown `aug` kwarg, so it raises `TypeError` before any data is touched. Combined with the two placeholder strings, the ACDC pipeline has demonstrably never been executed in this release. | [setup.py:56](../constraints/teds_net/dataloaders/setup.py#L56) vs [ACDC.py:24-27](../constraints/teds_net/dataloaders/ACDC.py#L24-L27) |
| B11 | **MNIST labels use a ~1/255 threshold, not the paper's 0.5.** Lines 47-48 map `>1 → 1` and `<1 → 0`, so the label is `pixel >= 1`: every faintly non-zero antialiasing pixel becomes foreground. Paper §4.1 specifies a threshold at 0.5. The resulting ground truth is a noticeably fatter blob than the paper's. | [mnist.py:47-48](../constraints/teds_net/dataloaders/mnist.py#L47-L48) |

### 2b. Present in the paper, absent from the release

- **No topology evaluation at all.** No Betti numbers, no `% |J_Φ| ≤ 0`, no Hausdorff distance.
  Every headline claim in the paper (100% topology preservation, 70% scene topology,
  the σ-sweep in Fig. 9) is unreproducible from this code as shipped. Only Dice is computed.
- **Dice is computed over the flattened batch**, not as a per-sample mean
  ([losses.py:11-24](../constraints/teds_net/utils/losses.py#L11-L24)). Not directly comparable to
  this repo's per-sample IoU without care.
- **Multi-structure segmentation is not in the release.** `out_chan` is 1 throughout, there is no
  multi-channel prior, and the paper's switch of final downsampling from MaxPool → linear
  interpolation for multi-class (§4.3) does not exist — only MaxPool
  ([TEDS_Net.py:68](../constraints/teds_net/network/TEDS_Net.py#L68)).
- **The paper's MNIST protocol is not in the release.** No Fourier augmentation;
  `GenPriorShape`/`SelectPrior` are defined but never called, so only digit "0" with a single
  prior is trained. The "all digits, paired priors, 99.53% topology" experiment (§5.1) needs writing.
- **The interpolation ablation (§5.2.1) is not parameterised.** `mode='bilinear'` is fixed in
  `mw_SpatialTransformer`; nearest/bicubic paths do not exist.
- **No checkpointing.** `checkpoint_freq` and `lr_sch` in the ACDC params are dead.
- **α/β are swapped relative to the paper's naming.** `loss=['dice','grad','grad']` maps to
  `output[1] = bulk` and `output[2] = ft`, so `weight[1]` is the paper's β and `weight[2]` its α
  ([trainer.py](../constraints/teds_net/trainer.py), `perform_losses`). Numerically irrelevant while
  α=β=10000; it bites the moment you sweep them separately.

### 2b-bis. Dead code — declared but never reached

Four things look like they implement paper features and do not. Worth knowing before you go
looking for them:

| Symbol | Where | Status |
|---|---|---|
| `params.net = 'teds'` | [mnist_parameters.py](../constraints/teds_net/parameters/mnist_parameters.py), [acdc_parameters.py](../constraints/teds_net/parameters/acdc_parameters.py) | Never read. `train_runner.py` hardcodes `from network.TEDS_Net import TEDS_Net`. |
| `UNet_MW` | [UNet.py:263](../constraints/teds_net/network/UNet.py#L263) | A complete U-Net baseline that nothing imports or instantiates. The paper's Table 1 U-Net row is unreachable from this code. |
| `GenPriorShape`, `SelectPrior`, `topo_dict` | [mnist.py:83-122](../constraints/teds_net/dataloaders/mnist.py#L83-L122) | The three-topology prior bank for digits 0-9. Defined, never called. `__getitem__` always returns the single annulus prior, and only digit "0" is loaded. The paper's §5.1 experiment (all digits, paired priors, 99.53% topology) is *not* in this release. |
| `betti: [1,1,0,0]` | [acdc_parameters.py:18](../constraints/teds_net/parameters/acdc_parameters.py#L18) | A config field for the expected Betti numbers. Nothing consumes it — there is no Betti computation anywhere in the repo. |

Also entirely absent, with no placeholder at all: image augmentation (the paper's ±5° rotation,
±5 px translation, flips, ±30% resize), Fourier augmentation for MNIST, subject-level split
logic, slice selection by topology, Hausdorff distance, perimeter extraction, checkpoint saving
(`checkpoint_freq` is declared and unused).

### 2c. Packaging blockers

- Missing dependencies: `raster_geometry`, `dataclasses_json`, `enforce_typing` (verified absent
  from `.venv`). `raster_geometry` is only used to draw circles for the prior — trivially
  replaceable with your own rasterizer if you'd rather not add the dep.
- Imports are absolute-from-root (`from network.UNet import ...`), so `constraints.teds_net.*`
  is **not importable as a package**. Works only when run with `teds_net/` as cwd.

---

## 3. Code explainer — the parts worth understanding before porting

### 3.1 Why the tanh activation is exactly "half a grid space"

[`DiffeoActivat`](../constraints/teds_net/network/utils_teds.py#L309) computes
`tanh(u) * (1/size[i])` in normalized units. Convert to pixels with the D3 factor:

```
(1 / size[i]) * (size[i] - 1) / 2  ≈  0.5 pixels
```

That is the paper's Eq. (2) bound — no grid point can cross its neighbour, so the *initial*
field is topology-preserving by construction. The neat part: `size` here is
`flow_field_size`, the **decoder-branch resolution**, not the input resolution
([utils_teds.py:49](../constraints/teds_net/network/utils_teds.py#L49)):

```
frac_size_change = [1, 2, 4, 8]           # indexed by dec_depth - 1
flow_field_size  = inshape / frac[dec_depth - 1]
```

For ACDC (`inshape = [144, 208]`):

| Branch | `dec_depth` | field resolution | half a grid space, in *fine* pixels |
|---|---|---|---|
| bulk | 4 | 18 × 26 | ≈ 4 px per composition step |
| ft | 2 | 72 × 104 | ≈ 1 px per composition step |

So the same bound gives the bulk branch coarse, large-displacement freedom (localise and
scale the prior) and the ft branch fine, small-displacement freedom (follow the contour).
That is the whole reason for two branches at two resolutions.

### 3.2 Scaling-and-squaring here is *amplification*, not integration — do not swap in `IntegrateVelocityField`

TEDS-Net's loop ([utils_teds.py:161-167](../constraints/teds_net/network/utils_teds.py#L161-L167)):

```python
for n in range(self.nsteps):
    vec = vec + self.transformer(vec, vec)   # Φ ← Φ ∘ Φ
    if viscous:
        vec = self.SmthKernel(vec)           # Gaussian smooth between compositions
```

Your `IntegrateVelocityField` ([modules.py](../constraints/voxelmorph/modules.py)) does:

```python
velocity_field = velocity_field * self.scale   # 1 / 2**steps   ← TEDS-Net has NO such line
for _ in range(self.steps):
    velocity_field = velocity_field + self.transformer(velocity_field, velocity_field)
```

VoxelMorph pre-divides so the result approximates `exp(v)` for a *fixed* velocity `v` —
magnitude is roughly invariant to `steps`. TEDS-Net deliberately omits that: it starts from a
half-pixel field and **amplifies** it, so `h=8` yields Φ^256 with displacements up to
~`2^8 × 0.5 = 128` grid spaces (this is Fig. 4 in the paper). Substituting
`IntegrateVelocityField` naively gives a field ~256× too small and the model simply cannot reach
the anatomy. Track B needs an `amplify=True/False` switch here, not a reuse.

The Gaussian smoothing *inside* the loop is the second TEDS-Net-specific bit — VoxelMorph has
nothing equivalent. It is what keeps interpolation-induced steep gradients (paper §3.1, Fig. 2)
from accumulating over the 8 compositions.

### 3.3 Grad loss is on Φ, not on u — a real divergence from your regularizer

TEDS-Net's `grad_loss` adds the identity grid *before* differencing
([losses.py:41-70](../constraints/teds_net/utils/losses.py#L41-L70)), so it penalises `∇Φ = I + ∇u`.
This matches the paper's Eq. (3), which is written on Φ. Your `DeformationGradientTerm` uses
`ne.nn.modules.SpatialGradient` on the field itself, i.e. `∇u` — the standard VoxelMorph
diffusion regularizer.

They are not the same objective. Driving `∇Φ → 0` pushes toward a *constant* map (collapse),
whereas driving `∇u → 0` pushes toward the identity. In normalized units the identity term per
step is only `2/(s-1)` (≈0.007 at s=288), but it is multiplied by β=10000. Worth flagging as a
hypothesis to test rather than a bug: it is a plausible contributor to the paper's own
observation (§6, Fig. 13) that TEDS-Net over-segments thin regions. Make `∇Φ` vs `∇u` an
explicit ablation axis in Track B.

### 3.4 The 2× field and the overlapping max-pool

`Mega_inshape = mega_P * inshape` ([utils_teds.py:51](../constraints/teds_net/network/utils_teds.py#L51)).
The composed field is upsampled to 2×, the prior is warped at 2×, and the *result* is
max-pooled back down with `MaxPool(kernel_size=3, stride=2, padding=1)`
([TEDS_Net.py:68](../constraints/teds_net/network/TEDS_Net.py#L68)).

Two things to notice:

- Halving grid spacing is one of the paper's three anti-folding modifications (§3.1, Fig. 2C) —
  the field better describes the transform, so fewer artefacts survive discrete storage.
- `kernel=3, stride=2` is an **overlapping** max-pool, i.e. a dilation. It thickens the mask by
  construction. This is a second, mechanical explanation for the over-segmentation in Fig. 13,
  independent of the smoothing explanation the paper gives. Cheap experiment: swap it for
  `kernel=2, stride=2` or the linear interpolation the paper mentions for multi-class, and see
  whether thin-region thickness improves.

### 3.5 grid_sample conventions

`mw_SpatialTransformer` ([utils_teds.py:172](../constraints/teds_net/network/utils_teds.py#L172)) does
`permute` then `new_locs[..., [1, 0]]`, because `grid_sample` wants `(..., ndim)` in **reversed**
spatial order (x, y) while the field is stored (y, x). Your
`vxm.spatial_transform` does the same thing with `.movedim(ndim_dim, -1).flip(-1)`. Identical
convention, different spelling — no conversion needed beyond D3's scaling.

### 3.6 Composition of the two branches

`forward` applies the ft field to the *already bulk-warped* prior, not to the original
([TEDS_Net.py:94-96](../constraints/teds_net/network/TEDS_Net.py#L94-L96)):

```python
_, flow_bulk_upsamp, bulk_sampled = self.STN_bulk(BottleNeck, enc_outputs, prior_shape)
_, flow_ft_upsamp,   ft_sampled   = self.STN_ft(BottleNeck, enc_outputs, bulk_sampled)
```

So the total transform is `Φ_ft ∘ Φ_bulk` and the topology guarantee rests on the composition
property (§3.1). The paper reports Jacobians per-field, never for the composition — a small,
honest gap you can close for free with your existing metric by emitting both fields as
`TransformSpec.steps` (`iter_deformation_fields` picks up both).

---

## 4. Task list

Tasks are sized at roughly 30–90 minutes. Each has an explicit **Done when** so you can stop
mid-phase without losing the thread. Phases are ordered; tasks inside a phase mostly are not.

### Phase P0 — get the vendored code running, unmodified (MNIST)

Goal: a known-good reference implementation to diff against later. Resist refactoring here.

- [ ] **P0.1** Add the three missing deps: `uv add raster_geometry dataclasses-json enforce-typing`.
      **Done when** `.venv/bin/python -c "import raster_geometry, dataclasses_json, enforce_typing"` exits 0.
- [ ] **P0.2** Run `train_runner.py` from inside `constraints/teds_net/` with cwd set there
      (imports are root-relative, see §2c). Do not fix the imports yet.
      **Done when** 20 epochs complete and a `Test Dice Loss` line is printed.
- [ ] **P0.3** Record the printed Dice against the README's `0.9272`.
      **Done when** the number is written into a scratch note next to the README value.
- [ ] **P0.4** Confirm bug **B1** in the running code: print
      `x.min(), x.max(), x.unique().numel()` for one training batch.
      **Done when** you have seen that the input has 2 unique values, i.e. it is the label.
      *This invalidates P0.3 as a segmentation result — that is the point of the task.*
- [ ] **P0.5** Fix only B1 (make `X_set` a real copy of the grayscale digits, `.copy()` before
      binarising `Y_set`), rerun, and record the honest Dice.
      **Done when** you have two numbers: degenerate-input Dice and grayscale-input Dice.
      Expect a meaningful drop; this is your true P0 baseline.
      Decide at this point whether to also fix **B11** (label threshold 1/255 vs the paper's 0.5).
      Fixing it makes the label thinner and the task harder — keep the two changes separate so you
      know which one moved the number.
- [ ] **P0.6** Write the delta from P0.5 into this doc as a note under §2a/B1.
      **Done when** the number is in the doc, not just your terminal.

### Phase P1 — instrument the reference (still vendored, still MNIST)

Goal: make the paper's headline claims measurable, using this repo's existing metrics.

- [ ] **P1.1** Write the normalized→pixel converter from §D3 as a small standalone function
      (`teds_field_to_pixels(field, size)`), with a unit test: a constant normalized field of
      `1.0` on a 29-wide axis must become `14.0` pixels.
      **Done when** the test passes. This function is the whole D3 interface; get it right once.
- [ ] **P1.2** Feed the converted `flow_ft_upsamp` from a trained P0 model into
      `DeformationJacobianTerm` and log `% |J| ≤ 0`.
      **Done when** you have a number for the σ=2 model. Paper predicts 0%.
- [ ] **P1.3** Sanity-check P1.2 by running the *same* field **without** conversion through the
      same metric.
      **Done when** you have observed it report ≈0% folding for a reason that has nothing to do
      with the model — the failure mode §D3 warns about. Write both numbers down.
- [ ] **P1.4** Write a 2D Betti-number helper: `betti(mask) -> (b0, b1)` using
      `scipy.ndimage.label` with 4-connectivity on the background for holes
      (the paper's §3.4 definition — note it is *not* 8-connectivity, and this choice changes
      results, cf. §5.2.2 on VoxelMorph's "holes").
      **Done when** it returns `(1,0)` for a disc, `(1,1)` for an annulus, `(1,2)` for a figure-8.
      Put it in `constraints/losses_metrics/` — it is generally useful, not TEDS-specific.
- [ ] **P1.5** Wrap P1.4 as a `MetricTerm` reporting "fraction of samples whose Betti numbers
      match the prior's".
      **Done when** it runs over a val epoch and emits one scalar.
- [ ] **P1.6** Re-run P0 with σ ∈ {0, 1, 2} and record `% |J| ≤ 0` and Betti-match for each.
      **Done when** you can say whether the σ→folding relationship of Fig. 9 reproduces.
      *This is the first result that is actually yours.*

### Phase P2 — port into the framework: `TedsLightning` on MNIST via your harness

Goal: TEDS-Net's numbers appearing in your normal W&B panels.

- [ ] **P2.1** Make `constraints/teds_net` importable: convert `from network.UNet import ...`
      to relative imports (`from .network.UNet import ...`) across the 5 files that need it.
      **Done when** `python -c "from constraints.teds_net.network.TEDS_Net import TEDS_Net"`
      works from the repo root. Keep everything else byte-identical.
- [ ] **P2.2** Replace the `dataclasses_json`/`enforce_typing` params objects with a plain
      frozen dataclass or a small config dict, so TEDS-Net can be constructed from your
      experiment args instead of a parameter module.
      **Done when** `TEDS_Net(cfg)` builds with a config you wrote by hand, and the two deps
      can be dropped from `pyproject.toml`.
- [ ] **P2.3** Write an MNIST dataset that emits your `Sample` TypedDict
      (`image`, `target_labels`, `sample_id`, `template`) — the prior goes in `template`.
      Grayscale input (not B1!), one digit class to start.
      **Done when** `PerSampleTemplateSource` accepts a batch from it without a `KeyError`.
- [ ] **P2.4** Decide and write the label schema for MNIST (background + digit).
      **Done when** `LabelSchema.label_map_to_one_hot` round-trips your `target_labels`.
- [ ] **P2.5** Write `TedsLightning` in `constraints/lightning_wrappers/modules.py`, modelled on
      `ProjectLightning` but per §1: no `spatial_transform` call, `segmentation_logits=None`,
      warped prior into `MetricInput.warped_template`, both fields (pixel-converted) into
      `TransformSpec(steps=...)`.
      **Done when** `--smoke_test` (`fast_dev_run`) completes one train + one val step.
- [ ] **P2.6** Build the metric composite for it: `RegistrationIoUTerm` +
      `DeformationJacobianTerm` + your P1.5 Betti term. Deliberately *omit*
      `SegmentationIoUTerm` (there are no segmentation logits).
      **Done when** a val epoch logs all three without a `None` crash.
- [ ] **P2.7** Port the two losses into `LossTerm` subclasses: a Dice term (per-sample mean, not
      batch-flattened — see §2b) and a `TedsFieldGradientTerm` implementing ∇Φ.
      **Done when** both appear as separate components in `LossResult.components`.
- [ ] **P2.8** Register a loss preset (`teds_dice_grad`) in `constraints/factories/losses.py`.
      **Done when** `create_loss_computer("teds_dice_grad", ls)` returns a working computer.
- [ ] **P2.9** Write `experiments/ex6/teds_baseline.py` modelled on
      [ex5/initial_decoupled_new.py](../experiments/ex5/initial_decoupled_new.py), with `--dataset mnist`.
      **Done when** a real (non-smoke) run appears in W&B with a `teds` tag.
- [ ] **P2.10** **Equivalence check.** Same seed, same hyperparameters: does `TedsLightning` on
      MNIST reach the P0.5 Dice?
      **Done when** the two numbers are within a tolerance you decide up front and write down.
      *If they diverge, the P0 reference is exactly what you diff against — this is why P0 exists.*

### Phase P3 — ACDC and artificial

With P2 done this should mostly be argument plumbing.

- [ ] **P3.1** Point the ex6 runner at `ACDCSliceMyocardiumOnlyDataset`; add the hollow-circle
      prior (r=35, thickness=7) as its template.
      **Done when** a smoke test passes on ACDC.
- [ ] **P3.2** Switch the config to the paper's ACDC hyperparameters (`net_depth=4`,
      `dec_depth=[4,2]`, `k=5`, `α=β=10000`, `T=0.3`, 200 epochs, batch 5, 144×208).
      **Done when** the config is asserted in the runner rather than remembered.
- [ ] **P3.3** Full ACDC run.
      **Done when** you can compare against the paper's Table 1 row
      (`Dice 0.85 ± 0.14`, `HD 4.82`, `100%` topology).
- [ ] **P3.4** Add Hausdorff distance as a `MetricTerm` (absent from both repos — see §2b).
      **Done when** it reports on the ACDC val set. `monai` is already a dependency.
- [ ] **P3.5** Point ex6 at `ComposedArtificialDataset` and run.
      **Done when** TEDS-Net numbers sit next to your ex5 runs on the same panels.
      **This closes Track A.**

### Phase P4 — Track B: extract the swappable components

Goal: an N-channel-in → field-out module usable wherever `VxmPairwise` is.

- [ ] **P4.1** Extract `HalfGridTanh` (§3.1) as a standalone module, parameterised by field
      resolution and unit convention (normalized or pixel).
      **Done when** a unit test shows max |displacement| ≤ 0.5 px for arbitrary input magnitude.
- [ ] **P4.2** Extract `SmoothedScalingSquaring` (§3.2) with an explicit
      `amplify: bool` flag — `True` reproduces TEDS-Net, `False` reproduces
      `IntegrateVelocityField`.
      **Done when** with `amplify=False` and smoothing off it matches
      `IntegrateVelocityField` output to floating-point tolerance. *That equality test is the
      whole point of the task.*
- [ ] **P4.3** Extract the super-resolution field + downsample stage (§3.4), with the max-pool
      kernel configurable.
      **Done when** `kernel=3,stride=2` (TEDS-Net) and `kernel=2,stride=2` are both selectable.
- [ ] **P4.4** Write `TedsFieldNet(in_channels=N) -> TransformSpec` — accepts image ⊕ mask
      concatenation, emits a pixel-unit field, satisfying the `registration_net` contract that
      `DeformableRegistrationNet` satisfies today.
      **Done when** it drops into `SegmentationRegistrationModel` in place of
      `DeformableRegistrationNet` and a smoke test passes.
- [ ] **P4.5** Add a `ProjectWithTemplateTeds` model class alongside `ProjectWithTemplateD`.
      **Done when** it is selectable from an experiment runner's `--modality`.

### Phase P5 — the comparison that the whole thing is for

- [ ] **P5.1** Fix the run matrix: {plain VoxelMorph, TEDS-inspired (P4.4), faithful TEDS-Net (P2)}
      × {artificial, ACDC} × ≥3 seeds.
      **Done when** the matrix is written down, with the metric that decides each cell.
- [ ] **P5.2** Ablate the anti-folding components independently on your best setup:
      σ ∈ {0,1,2} × activation on/off × 2× field on/off.
      **Done when** you have a table saying which modification actually buys the folding
      reduction *on your data* (the paper only shows this on ACDC).
- [ ] **P5.3** Ablate ∇Φ vs ∇u regularization (§3.3).
      **Done when** you can say whether the ∇Φ formulation contributes to over-segmentation of
      thin structures.
- [ ] **P5.4** Write up: which TEDS-Net ideas transfer, which do not, and why.
      **Done when** it is a section in `reports/`.

---

## 5. Deliberately out of scope

- Multi-structure / scene topology (paper §4.3, §5.3). Not in the release (§2b), and it is a
  separate research question from the folding machinery you actually want.
- 3D. The vendored code has 3D branches, several of them broken (B4, B6). Do not trust them.
- Reproducing the paper's Swin-Unet / TopoLoss baselines.
