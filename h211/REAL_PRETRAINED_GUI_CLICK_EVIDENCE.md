# H211 — first real public-pretrained visual GUI click-localization diagnostic

**Experimental, stacked Draft only.** Exact parent H210 PR #231 commit
`6fb504f2cf4914cfaf492fcf960943b9bc7a2c88`.
No released Hala model, no private media or gold, no OOD, no training.

## Why this is different from H209

H209 performed two real **color recognition** calls on self-generated
plain color panels. Its public 256M model answered RED correctly and
BLUE incorrectly (1/2), but there was no GUI command emitted,
source-relative coordinate, or action parser.

H211 tests the more relevant contract: on a **completely original
self-authored** 960×720 application-like raster, can the same
public pretrained model emit a **strict one-action native JSON click**
on a blue SAVE button rather than a gray CANCEL button?

Source: `h211/public_gui_grounding_baseline.py`. The same task text
and two distinct dynamically painted positions (LEFT/RIGHT) are
supplied to the real vision processor. The coordinate ground truth
is in the source-side grader only: it is never inserted into the
model-visible prompt. The raster and generated response are transient
memory objects only. Source UI contains no private user screenshot,
no browser, no DOM, no borrowed old candidate image, and no gold file.

Rubric: `parse_actions` from H199/H200, one action only, exact JSON,
finite bounded 960×720 coordinates. Only a `click` landing inside
the BLUE SAVE rectangle counts as HIT. Clicks elsewhere are MISS;
unparseable responses MALFORMED; abstention ABSTAIN; other action kinds
NONCLICK. **No heuristic coordinate extraction**, forced fallback
action, or interpreting words like 'left' as a successful click.

This evaluates a **distinct unmodified public 256M model**, not
Hala's Qwen3-VL/LoRA policy. A 2-case self-generated diagnostic
does not estimate held-out original app performance, bilingual
capability, true GUI control generalization or official OOD. Zero
native browser events. Native action *dispatch*, if safely authorized,
continues to be governed separately by H207/H208.

## Real owner environment, immutable source and operating limits

- Public pretrained model `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`,
  immutable Hugging Face revision
  `9b299836fe84e0aabd39e1e07651e569e5dfe87c`,
  Apache-2.0, 1,025,998,224 safetensors bytes. No model donation to Hala
  training or model redistribution.
- Owner Windows 11 Intel CPU; no NVIDIA GPU. A dedicated throwaway
  `D:\Hala-H211-QA` venv with
  PyTorch CPU 2.7.1, torchvision CPU 0.22.1, Transformers 4.57.3,
  Pillow 12.3.0, Accelerate 1.11.0, and num2words 0.5.14.
  Explicit isolated `HF_HOME`, minimum available disk space of 3GiB,
  fixed revision, `trust_remote_code=False`, deterministic
  `generate`, max 64 response tokens. No paid service or auto remote
  user-media upload.
- Only public-source/generated synthetic tests run on free Linux CI,
  with `HF_HUB_OFFLINE=1`. No hosted model weights/GPU/CPU inference.

## Observed results

Owner Windows CPython 3.11 synthetic-only tests ACTUALLY PASS:
- H211 original-scene and strict point-grading: **10/10**, normal and
  optimized `python -O`.
- H209 inherited original-source/public model contract: **7/7**,
  normal and optimized.
- **17 tests per mode, 34 executed passes total**, no skips.
- `compileall` passed.

**Actually observed real public model Windows CPU run after tests:**
the frozen public model and HF processor loaded successfully into CPU
from the pinned revision, and **two real model.generate calls**
processed the two fresh 960×720 self-authored GUI scenes. The same
prompt was used for LEFT and RIGHT with no coordinates supplied.
**Both** generations emitted **45 new tokens**; the actual vision
tensor exposed **10,223,616 elements per call**.

The strict original H199/H200 action parser rejected **both
outputs as MALFORMED**. The true result is **HIT 0/2, MISS 0/2,
MALFORMED 2/2, ABSTAIN 0/2, NONCLICK 0/2**. The result is **not**
a successful GUI click baseline: there were **zero native input
events**. Exact observable terminal receipt:
`H211_PUBLIC_REAL_GUI_CLICK_DIAGNOSTIC_COMPLETE
model_calls=2 hits=0 misses=0 malformed=2 nonclick=0 abstain=0
native_browser_events=0 hala_optimizer_steps=0 official_ood_cases=0
private_media_opened=0 elapsed_seconds=158.4`.

The 158.4 seconds include loading model weights and inference for
both cases on the CPU. No user images, old OOD or owner source cases
were read. This is model-use **failure evidence**, not evidence of
Hala improvement. Under this exact strict action contract, do not
promote the untrained 256M public model into any browser input
executor. Do not publish raw generated model text or screenshots.

## Separate unsolved model goals

- Qualified base Qwen3-VL processor/model exact weights and genuine
  GUI action policy execution on original source.
- Independently evidenced source/fork history and old decoded RGBA
  exclusions, official role-disjoint TRAIN/DEV admission; H199 approved
  TRAIN allowlist **EMPTY**. All 20 current owner original-GUI
  prospects remain unadmitted.
- New independent pristine OOD: H173 consumed one-shot OOD and
  reserved Gantt remain off-limits.
- No model fine-tuning or official accuracy claim from this diagnostic,
  no merge to protected main, release or publication.
