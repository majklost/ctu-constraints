# Metrics, overlays, and reporting refactor checklist

This is the working checklist for replacing the legacy per-batch metric and
W&B-overlay path with stateful metric terms, standalone overlay computers, and
a logging provider.

## Decisions already made

- `MetricInput` contains semantic model/batch data only.
- `StepContext` contains Lightning lifecycle data only.
- `MetricResult.scalars` is always a dictionary; an empty result uses `{}`.
- Stages use `"train"`, `"val"`, `"val_extra"`, and `"test"`.
- `OverlayResult` is the generic overlay payload; no separate `ImageOverlay`
  type is needed.
- `OverlayPolicy` is immutable.
- Overlay `sample_ids` identify stable samples globally, rather than positions
  within whichever batch happens to be processed. `MetricInput.sample_ids`
  carries them in batch order.
- Overlay computers are enabled by providing them. An empty overlay-computer
  collection means no overlays; policies do not use `None` or empty stages as
  an implicit "all stages" setting.

## 1. Core contracts

- [x] Remove lifecycle fields from `MetricInput`.
- [x] Introduce `StepContext`.
- [x] Use an always-present `MetricResult.scalars` dictionary.
- [x] Standardize the stage name `train`.
- [x] Make `OverlayPolicy` immutable.
- [x] Validate `OverlayPolicy` stages, frequency, and unique sample IDs.
- [x] Carry stable dataset IDs as `MetricInput.sample_ids` and resolve them in
  O(batch size + configured sample count) time.
- [x] Keep `OverlayResult` free of W&B-specific types.

## 2. Stateful metric terms

- [x] Add the stateful `MetricTerm` API: `update`, `compute`, and `reset`.
- [x] Implement stateful segmentation IoU.
- [x] Implement stateful registration IoU.
- [x] Harden composite-result merging and duplicate-name checks.
- [x] Implement `SegmentationConstraintViolationTerm` with accumulated
  violating-sample and total-sample state.
- [x] Implement `RegistrationConstraintViolationTerm` with the same behavior.
- [x] Ensure terms handle absent optional model outputs by returning empty
  batch results without changing inappropriate state.

## 3. Composition and stage routing

- [x] Make `StagedMetrics` an `nn.Module` so its child torchmetrics state is
  registered, moved between devices, checkpointed, and usable in distributed
  training.
- [x] Rename it to `StagedMetricComputer` if naming should match the design.
- [x] Keep its per-stage composites in `nn.ModuleDict`.
- [x] Provide separate term instances for every stage; never share a train
  term with validation.
- [x] Add an explicit no-op stateful metric for intentionally empty stages.
- [ ] Create the default composition/factory for train, validation, extra
  validation, and test as applicable.

## 4. Logging provider

- [x] Implement `LightningWandbLoggingProvider` using only the bound scalar
  logging function (`scalar_log=self.log`).
- [x] Add batch-scalar logging.
- [x] Add epoch-scalar logging.
- [x] Centralize standard names, with stage first, for example
  `val/epoch/segmentation/iou/pred_vs_gt`.
- [x] Look up the W&B run lazily when logging artifacts.
- [x] Convert `OverlayResult` to `wandb.Image` inside the provider only.
- [x] Safely no-op for no trainer, non-W&B loggers, and nonzero ranks.

## 5. Lightning wiring

- [ ] Remove `MetricLoggingMixin` and all legacy `MetricResult.logs`,
  `sum_logs`, and `wandb_overlays` handling.
- [ ] Replace imports and constructor types for the removed metric-computer
  classes.
- [ ] Add one small `_step_context(stage, batch_idx)` method to each common
  Lightning base/shared implementation, returning:

  ```python
  StepContext(
      stage=stage,
      batch_idx=batch_idx,
      current_epoch=int(self.current_epoch),
      global_step=int(self.global_step),
  )
  ```

- [ ] In each shared step, build `MetricInput`, obtain the context, call
  staged-metric `update`, and delegate any batch result to the provider.
- [ ] At each relevant epoch end, call `compute(stage)`, log the epoch result,
  then `reset(stage)`.
- [ ] Ensure `val_extra` has independent update, compute, and reset handling.

## 6. Overlay computers

- [ ] Define the `OverlayComputer` abstract contract.
- [ ] Implement `SegmentationOverlayComputer`.
- [ ] Move all remaining overlay computation out of `metric_computers.py`.
- [ ] Filter by stage and epoch frequency with `OverlayPolicy` and
  `StepContext`.
- [ ] Resolve requested global sample IDs to samples in the current input via
  `OverlayPolicy.batch_positions(metric_input.sample_ids)`.
- [ ] Produce generic `OverlayResult` payloads containing image, masks, class
  labels, and caption.

## 7. Callers and compatibility

- [ ] Update experiments that construct `DefaultSegmentationMetricComputer`.
- [ ] Update early-stopping callback keys if epoch metric names change.
- [ ] Delete the commented legacy metric/overlay implementation.
- [ ] Remove all construction of lifecycle fields inside `MetricInput`.

## 8. Tests

- [ ] Update legacy metric-computer tests to use `update -> compute -> reset`.
- [ ] Test IoU across unequal batches to prove dataset-level accumulation.
- [ ] Test state isolation across train, val, and val_extra.
- [ ] Test constraint violation accumulation and reset.
- [ ] Test composite duplicate-name errors.
- [ ] Test overlay policy filtering and global sample-ID resolution.
- [ ] Test `OverlayResult` contents without importing W&B.
- [ ] Test logging-provider metric names and W&B conversion using mocks.
- [ ] Add Lightning integration tests for batch logging and epoch-end reset.
- [ ] Run `.venv/bin/pytest -q`.
