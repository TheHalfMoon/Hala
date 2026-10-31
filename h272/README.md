# H272 — Joint visual-language LoRA and native target-hit research

**Source-only public code. Owner-private FREE Nvidia T4 training. No model release.**

## Scientific reason

The actual H271 first-party private T4 experiment completed **8/8 real
vision-only LoRA optimizer updates** on the pretrained pinned
Qwen3-VL-2B-Instruct, with exactly 16 visual adapter tensors and no
language adapter. Yet both original Arabic Stockroom tasks FAILED:
base 0/2 and vision-only adapter 0/2; the independent, after-prediction
first +1 button bounding-box hit scorer showed **0/2 for both**. H269
language-only LoRA (2 real steps) and H270 language-only weighted LoRA
(8 real steps) also had **no task-state improvement** on this single
author-owned first-party GUI family.

H272 tests **a bounded, prospective JOINT vision+language LoRA** instead
of merely repeating those single-modality pilots. It reuses the verified
H106 trainable-name gate from the Hala trainer:
- Exact **8** fused vision QKV/output attention Linear modules, last
  four vision blocks 20–23; **16** corresponding LoRA A/B tensors.
- Only actual Qwen3-VL language_model q_proj/v_proj Linear modules;
  bounded 16–256 modules, exactly **2 A/B tensors per module**.
- Frozen pretrained visual/language base weights, only matched LoRA
  tensors allowed to update, a bound of 8 million trainable values.
- Original Hala completion-only supervised labels and 8x coordinate
  digit weights. Eight actual FP16 rank-4 optimizer steps, LR 5e-5,
  conservative previously qualified GradScaler init_scale=1.0.
  Every step must show a finite nonzero gradient, supervised coordinate
  tokens and genuine adapter changes, or abort without claiming
  qualification.
- A paired comparison on the **two same previously exposed first-party
  Arabic RTL original Stockroom scenes**. The model sees screenshot
  pixels and a task prompt only, not source, DOM, gold or any evaluator
  geometry. Original H208 model vision inference and H268 token-time
  exact H199 native JSON grammar; H264 screenshot-staleness guard;
  H263 real native browser actuation. After prediction/action only,
  the independent original DOM scorer reports whether the click
  landed inside the correct first +1 button and whether stock quantity
  actually increased.

**All H271/H272 first-party scenes were already inspected in research.**
This is diagnostic A/B on an exposed single original family, NOT
independent blind OOD, NOT newly source-disjoint TRAIN/DEV, and NOT
evidence of generalizable accuracy even if a click succeeds.
There is no retrospective fence stripping, DOM oracle to policy,
posthoc target injection, gold fallback or new H173 access.

## Source, identity and privacy gates

Read original H271 public source only from exact frozen SHA
`3e53eca0991b1032ceafdb20fd28e277e8032a98` on branch
`ps/h271-vision-lora-private-native-diagnostic`, ensure clean Git
and pinned author-owned H262 HTML SHA256
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
Qwen/Qwen3-VL-2B-Instruct model pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203`.
Transformers 5.16.1, PEFT 0.20.0 and the pinned MIT
lm-format-enforcer core source commit
`817f944fcc8851917c40300a47f62d1defc3ffc3` required.
Original Kaggle CUDA PyTorch is never replaced; only already
audited optional incompatible torchao 0.10.0 is removed within the
private ephemeral Kaggle environment.

**Kaggle code-only**, is_private=true, free/noncommercial NvidiaTeslaT4
only, no uploaded model/adapter/datasets/source media/weights/gold,
no paid inference, no deployment. Qwen weights, original app, training
PNG/labels and adapter/optimizer all in Kaggle
`/tmp/hala-h272-firstparty`, deleted at end. Only one tiny private
aggregate JSON in `/kaggle/working` is allowed:
`hala-h272-private-joint-native-aggregate.json`. It reports coarse
genuine optimizer, validated module partition, model generations,
native dispatch, correct-first-button bounding-box hits, and actual
inventory task outcome, with explicit false source-independent OOD
and trained-generalization fields; NO screenshots, model completion
text, coordinates, gold or weights.

Never run unfiltered `kaggle kernels output` (would risk huge model
weight downloads). Only after COMPLETE, fetch exactly the above JSON
with `--file-pattern`, inspect the private aggregate, delete its
owner-Windows scratch copy, and publicly document the aggregate only.

### Reproducibility

Files:
- `h272/joint_lora_native.py`: actual pretrained and joint adapted
  private Kaggle native browser pilot, fail-closed
- `h272/build_private_payload.py`: immutable code-only private
  free-T4 Kaggle packaging
- `tests/test_h272_private_joint_native.py`: synthetic CPU-only
  bounds, true joint layer pattern, privacy and no-fake-accuracy gates

To create the private upload using already owner-authorized Kaggle CLI:

    python -m h272.build_private_payload --source h272/joint_lora_native.py --destination D:\Hala-H272-Private-Upload
    kaggle kernels push -p D:\Hala-H272-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h272-private-joint-lora-native-t4

Delete this-turn upload scratch after accepted. No local model weights.

## Unchanged scientific barriers

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY. H173 original
32 scored cases permanently CONSUMED/SEALED; H212 scored procedural
rows barred. H250/H251 original other-app media research quarantined
pending transitive visual assets/forks/npm/history and PNG plus decoded
RGBA duplicates and rights. Author-owned H262 is NOT independent
source-disjoint OOD. A genuine Hala release requires lawful
multi-original-family TRAIN/DEV, new never-exposed blind source-heldout
OOD, paired real native task-state improvement and release safety
qualification. Do not publish model or make accuracy claims without it.
