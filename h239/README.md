# H239 — bilingual Qwen3-VL processor alignment for `type_text`

**Source/test-only research. No Hala model weights, original owner GUI images,
historical H173 OOD, gold, training, new DEV or official OOD evaluation.**

## Actual action/model gap

H216 verified a pinned original Qwen3-VL 2B processor with original
screenshot-native `click` actions, but H234 added screenshot-grounded
`type_text` actions (English and Arabic) without confirming multimodal
prefix and assistant-completion alignment for that new action vocabulary.

H239 provides a **separate opt-in processor-only verifier** for two
self-authored English/Arabic action strings. It uses a deterministic
960×720 in-memory PIL synthetic screenshot. The actual tokenizer/vision
processor, if already installed, must independently establish:

- Identical prefix token IDs when one assistant action is appended
- Identical vision pixel tensors and image-grid tensors
- Loss masks `-100` for all user, prompt, and image-prefix tokens
- Assistant completion decodes precisely to a strict valid bounded
  `type_text` action, including Arabic text preservation
- No reference action text hidden in the visible task prompt
- No prohibited control characters, bidi-format injections, DOM
  selector targeting, multi-action macros or unsafe coordinates

The verifier REUSES H199's existing assistant-only loss-mask primitive,
H216's pinned processor byte checks and tensor comparison, and H234's
exact v4 text-action parser. It does **not** expand H199's empty
`ADMITTED_TRAIN_MANIFEST_SHAS`.

## CPU unit tests

```powershell
python -m pytest -q tests/test_h239_bilingual_processor_parity.py tests/test_h234_native_type_text.py tests/test_h199_native_action_sft.py
python -O -m pytest -q tests/test_h239_bilingual_processor_parity.py
```

The CI suite uses an intentionally **fake** processor for policy-failure
mutations and cannot prove real Hugging Face tokenization. A real
offline-only run requires a preexisting approved environment with
`transformers`, `torch`, Pillow, and the exact cached Qwen3-VL processor
revision `89644892e4d85e24eaac8bacfd4f463576704203`. Run with
network explicitly disabled:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -m h239.bilingual_processor_parity --local-processor-snapshot <PATH_TO_ALREADY_CACHED_PINNED_PROCESSOR>
```

The source entrypoint fails CLOSED when the processor or one of the
H218 8 pinned processor/tokenizer assets is unavailable. It never
downloads model weights, invokes a model, prints private text or trains.
H239 does not provision a new venv or download large dependencies.

**Current owner Windows result:** the pinned snapshot exists, but the
selected local Python environment lacks `transformers`, so the real
processor stage could NOT be qualified from that environment. This is
an explicit blocker, NOT an observed pass. The previous H216
processor-only click result is not evidence of H239 type_text parity.

## Scientific blockers unchanged

H238's public minimum historical source floor has 27 old donor
families; the owner-only H178 inventory also found older locally
present source families not represented there. The scoped H168/H171
legacy TRAIN manifests have old PNG hashes but no decoded RGBA SHA
fields. Full historical Git/fork/npm/visual registry and transitive
media rights remain UNPROVEN. H199 officially admitted TRAIN manifests
remain EMPTY; H212 research scenes cannot receive optimizer gradients,
and H173 original scored OOD is permanently consumed and sealed.

No real TRAIN/DEV admitted, no Qwen LoRA weights or model improvement,
no merge, main push, model publication, or paid service.
