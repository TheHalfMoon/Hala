# H54 — Real Original DragOn Native Ordered-Drag LoRA Source Bridge

**2026-10-21. The next model-learning gate is real remote gradient optimization, NOT fake click labels.**

## Measured and source-qualified

The H53 genuine original `Hcompany/DragOn` official `train/slider.tar` shard (commit `3451c1fc3543c18fc3361a32bfe1f12cc69b64b5`) has 87 distinct original TRAIN JPEG screenshots and 760 original start/end drag annotations. H53 original raw JSONL SHA256 is `9abab62624b45ead5db1ed0c1fdb5d7123f7c761c0a32b8a35f9ecfdebc96c09`. H54 ran its new source converter **against all genuine 87 original JPEG + original JSON sources and H53 signed SHA receipts**, producing a real **760-row Qwen3-VL SFT candidate** whose immutable derived JSONL SHA256 is `205b408d314576d389fb497530fc44d115208b07d2b40501387c585c0700d40b`.

Crucially, the model target is **a true ordered drag**, never a click:
```json
{"coordinate_space":"source_image_pixels","end_point":[675,580],"ordered":true,"start_point":[372,580],"type":"drag"}
```
The sample above is the observed first original TRAIN label of the actual source conversion; **it is not a neural prediction or a model success**. The textual instruction explicitly binds these pixels to the original screenshot dimensions, so processor-resize does not silently redefine target coordinates.

`h54/native_drag.py` replays the entire private original source audit, verifies signed H53 per-JPEG and per-task SHA256 against immutable source UUID and TAR offsets, enforces all 760 TRAIN-only ordered drags and preserves the exact 87-image media identity. It creates only separate `h54_drag_sft.jsonl` and `h54_drag_training_manifest.json` *inside a private source directory*, leaving original Kaggle H53 JSONL, manifest, 87 JPEGs and JSONs unchanged.

`training/scripts/train_adapter.py` now dispatches `hala.h54.native-drag-sft.v1` only after this rigorous source verification. It permits **original JPEG image preprocessing solely for verified native H54 drags**, during both the full pre-GPU source audit and each eventual gradient microbatch. Previous PNG-only H48/H49/H50/H52 click paths are unchanged and retain their strict format checks. The Qwen3-VL trainable-parameter policy, version pinning, checkpoint/resume, GPU model fit, cost, deterministic source freezing and dataset provenance checks remain unchanged.

## Actual current Kaggle and source access

**Approved PRIVATE dataset only:** https://www.kaggle.com/datasets/abdulazizshehri/hala-gui-training-library-private (ID `12752110`; private authenticated listing, unauthenticated HTTP 403 verified in H53). Original H50/H52 CC0 train and genuine H53 Hcompany/DragOn Apache-2.0 TRAIN-only source retained. Do not re-upload or re-license the original H53 screenshots, do not use deleted historical public slug. Public GitHub receives only code, synthetic test fixtures generated at runtime, traceable non-sensitive source hashes and frozen documentation. **H54-derived data are temporarily generated locally and are NOT yet uploaded as a separate Kaggle version**. Because H54 is deterministic, a future approved free Linux CUDA session can rederive the exact SFT JSONL from the preserved original private-source ZIP.

## Reproduce source admission on authorized *private* source files

```bash
# AFTER downloading/extracting the authenticated PRIVATE Kaggle
# hala-h53-dragon-slider-original-train archive into /tmp/hala/h53-train:
python -m h53.dragon_train_shard audit --output /tmp/hala/h53-train
python -m h54.native_drag --source-private-train-dir /tmp/hala/h53-train
python - <<'PY'
from pathlib import Path
from training.scripts.train_adapter import load_authoring_records
from training.artifacts import strict_json
root=Path('/tmp/hala/h53-train')
manifest=strict_json((root/'h54_drag_training_manifest.json').read_text())
rows=load_authoring_records(root/'h54_drag_sft.jsonl',manifest)
assert len(rows)==760
assert all('"type":"drag"' in row['completion'] for row in rows)
print('Native drag model source-admission records:',len(rows))
PY
```

**Model run is BLOCKED until genuinely approved FREE Linux CUDA and real unchanged-model drag BASELINE evidence.** The candidate runner requires `HALA_REMOTE_TRAINING_ALLOWED=1` in explicit approved Linux, rejects Kaggle compute under previous provider terms qualification, and forbids local model weights. Additionally, H54 requires an independently sourced **at least 16-case app-disjoint original drag baseline** from the exact frozen Qwen3-VL model, with source SHA256 and observed deficit. The currently named `held_out_families` entry is explicitly a *reservation*, **not evidence that such a dataset exists**. H46 0/8 click selection does not prove a native drag baseline; do not repurpose it as one. No actual GPU model, gradient update, LoRA adapter, loss curves, trained weights or independent drag task success exists. The model benchmark gate remains **NOT DONE**.

## Actual test requirements and status

- Local CPU original-source H54 conversion ran: 760 real TRAIN rows from 87 originals; signed source hashes replayed. No MODEL run.
- `tests/test_h54_native_drag_sft.py`: first-party fake JPEG/JSON fixtures only, no DragOn media committed. Tests include original source verification, modified image/task refusal, duplicate/modified dataset refusal, non-overwriting outputs, native-action (not click) schema, JPEG opt-in vs PNG legacy path, and real-baseline schema failure cases. Initial test iteration failed due to Python default-argument snapshot semantics under synthetic monkeypatch, fixed in the new branch; the initial failure is not model failure and source is unchanged.
- Exact full source unit suite/Graft/GitHub CI and independent D1/PStack semantic reviews are separate qualification stages: do not claim them until actual reports exist.

## Signed source-engineering qualification (2026-10-21)

- **8/8 H54 focused Windows CPU tests PASS**, original `reviews/h54-native-drag-focused-20261021.log` SHA256 `a5d483b41fd0eeb72f60965a7ac13ac79435e2d004b1991225ea34490978ac13`. Early source-test red after introducing H54 default receipt overrides was **repaired**, not suppressed, and final tests use only first-party fake runtime JPEG fixtures; no real Hcompany screenshots were placed in this public GitHub tree.
- **Full Windows source `unittest discover`: 652 total, 650 PASS, 2 existing SKIP, zero failures, 221.419 seconds**. Original `reviews/h54-native-drag-full-suite-20261021.log` SHA256 `ab358e6c691f27a7802b488ecc2eababb566de621702a7cf67942a3f71c773c5`.
- Genuine **Graft wiring graph build and check PASS** on 248 files, 2,022 nodes and 5,421 edges; `reviews/h54-native-drag-graft-build-20261021.log` SHA256 `f2f7474678ce8f2c408e66d5088354fd02bb1e1907fc6560ed0c0ea3fd6e92ed`, check log `reviews/h54-native-drag-graft-check-20261021.log` SHA256 `3441ccb3baa9a1cc461dadd696838bcd7c24c4907bcbaa8b61ffde2491482684`. Graft semantic deep layer **NOT** run.
- D1, PStack and Alibaba independent semantic model review **NOT** run. Only genuine CPU audit and real Graft wiring were performed. **Hosted GitHub Python 3.11/3.12 exact-head CI must still be checked** after the signed commit; no hosted claim is made from these local logs.
