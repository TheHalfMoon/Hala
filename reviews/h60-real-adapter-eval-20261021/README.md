# H60 — real gradient training and the matched adapter evaluation, 2026-10-21

This bundle holds the non-sensitive receipts of the first **real, gradient-bearing**
Hala training run and of the **matched** comparison between the frozen unchanged base
model and the trained adapter on the first-party PUBLIC DEV drag scenes.

Nothing here is a mock, an author-oracle harness, a proposed experiment or a vendor
score. Every number was produced by a real forward and backward pass on a real GPU
and is bound to an immutable digest.

## What ran

| Fact | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct`, pinned revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| Runtime | free Google Colab **Tesla T4**, `colab usage` balance **0.00 compute units** before and after |
| Frameworks | torch `2.11.0+cu130`, transformers `5.16.1`, peft `0.20.0` |
| Training source | the verified private original DragOn TRAIN export, dataset SHA `10cdcee91ae219e45f74850a165cb1ce1228d7fd45f8b1b2274b5e69f934f99a` |
| Checkpoint | `step-00000060-8647e997c44f4dc3acf5220efce56bea`, checkpoint SHA `c166d318fcf8a5f996aab8b453aa358b8d8d48ba714a5e6c27311c5f280397b4` |
| Adapter weights | `adapter/adapter_model.safetensors` SHA `d0e594035ccf912e7dffa4a96025ba16a0215e2fcc8acdf60d34b462cce69249` |
| Resume state | `resume.pt` SHA `1dce7035026e3b74cfc4d58c427249336001028c0b5a70b37f3e657cd10e4674` (optimizer, scaler, torch and CUDA RNG state) |
| Changed adapter parameter tensors | **112** |
| Training loss | **2.4253 → 0.5490** over 60 optimizer steps (1,127.9 s of measured step time) |
| Gradient norm before clipping | 7.2612, 7.3768, 7.1443, 6.7758, 6.2126, 6.2878, 5.4314, … — **finite and nonzero at every recorded step** |
| Maximum parameter update per step | ≈ 5.0e-05, consistent with the 5e-05 learning rate and the unit-norm clip |
| Peak allocated VRAM | 8.64 GB of 15.64 GB |
| Trainable / frozen parameter residency | adapter 1,605,632 params in float32 on `cuda:0`; frozen base 2,127,532,032 params in float16 on `cuda:0` |

The training loop refuses to continue on a non-finite loss, a non-finite gradient or a
zero gradient norm, and it refuses to publish a checkpoint that records no changed
adapter parameter tensor. The run finished with exit code 0 and its own
`training-result.json` records `adapter_qualified: false`, `evaluation_required: true`.

## The matched comparison

Both runs use **the same runner** (`h60.real_drag_dev_baseline.run`), the same case
loader, the same prompt text, the same prompt field allowlist, the same screenshot
provenance checks, the same published-answer leak guard, the same generation settings
and the same scorer. The only difference is the model.

| PUBLIC DEV set | Metric | Frozen unchanged base | H60 adapter (60 real steps) |
|---|---|---:|---:|
| H55 bilingual drag (24 scenes) | strict-valid raw outputs | **0 / 24** | **14 / 24** |
| H55 | both endpoints hit | 0 / 24 | 0 / 24 |
| H58 mixed-domain drag (24 scenes) | strict-valid raw outputs | **0 / 24** | **3 / 24** |
| H58 | ordered semantics valid | 0 / 24 | **3 / 24** |
| H58 | geometrically correct | 0 / 24 | **0 / 24** |

## The honest reading

* The adapter **real and reproducibly changed the model's output format**: the frozen
  base never emitted a strictly valid bare-JSON drag on either set, and the adapter
  emits 14/24 on H55 and 3/24 on H58. Ordered semantics recovered on 3 H58 scenes.
* The adapter **did not yet solve grounding**: it is still 0/24 geometrically correct
  on both sets, so **real GUI drag task success is still 0** on the measured sets.
* The H55 gain is English-only: 12/12 valid on the English scenes and 2/12 on the
  Arabic scenes, which is consistent with the training corpus being English original
  source. Arabic grounding is not addressed by this run.
* Therefore: **no adapter qualification, no benchmark comparison, no blind test and
  no "best model" claim is made.** The next measurable target is geometric grounding,
  which needs a curriculum that supervises the endpoint coordinates and an Arabic
  share of the source, not more of the same English drag supervision.

## Files

| File | Content |
|---|---|
| `summary.json` | the real training metrics and the matched comparison |
| `h55-adapter-run-result.json`, `h58-adapter-run-result.json` | the full adapter inference receipts, including the verified checkpoint identity and the per-step training qualification evidence |
| `h55-frozen-run-result.json`, `h58-frozen-run-result.json` | the frozen-base receipts for the same protocol |
| `*-raw-attempts.json` | the raw model output strings, verbatim, for every scene |
| `*-frozen-score.json`, `*-adapter-score.json` | the output of the unmodified H55/H58 scorers |
| `manifest.json` | the byte size and SHA-256 of every file in this bundle |

No model weights, no credentials and no third-party original media are stored here.
The adapter weights and the resume state remain in the private training runtime.
