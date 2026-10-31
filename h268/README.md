# H268 — model-time constrained native JSON (private free T4 research)

**Research only.** This is **not** trained Hala, new independent OOD,
officially admitted TRAIN/DEV/OOD, a benchmark gain or a release.

## Scientific trigger and preregistered hypothesis

The H267 private owner Kaggle run **completed two genuine pretrained
Qwen3-VL-2B-Instruct multimodal generations** on H262's original first-
party Stockroom app. Both unrestricted prompt variants returned
`MARKDOWN_FENCE` and were rejected by H199, causing **zero native
actions and 0/2 real GUI task-state successes**. Stricter prompting
alone failed; those old responses were never persisted. Do NOT strip
fences retrospectively or use model-invisible reference geometry.

H268 tests **one prospective, preregistered, token-filtered pretrained
generation** on the *same first-party research task*, using:
- Official Transformers `generate(prefix_allowed_tokens_fn=...)`,
  with greedy decoding and a fixed 160-token maximum as in H208.
- MIT `lm-format-enforcer==0.11.3` (fixed version) to map an explicit
  bounded native-action regular expression to allowed tokenizer IDs
  at **each generation step**. This is not post-hoc JSON sanitization.
  Upstream: https://github.com/noamgat/lm-format-enforcer
- The grammar allows exactly TWO families: complete
  `{"actions":[{"kind":"abstain"}]}` OR a complete
  `{"actions":[{"kind":"click","point":[x,y]}]}`, where **the model
  chooses both coordinates** (x 0–959, y 0–719), not supplied by gold
  or DOM. No other actions, markdown, extra keys, leading zero,
  negative or out-of-bounds coordinates.
- Exact H199 parser is STILL invoked by the existing H208 policy.
  H264 verifies the exact model-visible screenshot remains unchanged
  before passing any strict native action to H263 real Playwright.
  `abstain` is explicitly preserved as safe no-op.
- One fresh offline browser context, one original H262 first-party
  research scene, one pretrained model call, then an independent
  first-item inventory increment oracle used **only after** inference.
  No old H173 scored OOD, H212 barred rows, private original app media
  or reference coordinates are read by the model.

The *format* may become valid without visual *grounding* improving.
The only task reward is an actual unit increase in the original app.
Even successful formatted generation + stock increment would be
**first-party single-case pretrained research**, not learned skill
or independent/generalizable OOD gain. Log actual failures and refusals.

## Reproducibility, cost and safety

The kernel uses owner-connected Kaggle as an **is_private=true**
script requesting only **FREE noncommercial NvidiaTeslaT4**, no
attached media/datasets/weights or credentials, with 8GiB ephemeral
space prerequisite. The existing Kaggle CUDA Torch is never replaced.

Pretrained Qwen/Qwen3-VL-2B-Instruct revision
`89644892e4d85e24eaac8bacfd4f463576704203` and Transformers
`5.16.1` are pinned. Model/processor weights are downloaded
**in the remote ephemeral environment** to `/tmp/hala-h268-research/hf`,
with explicit `cache_dir` and `HF_HOME`; the exact original public
source is cloned under `/tmp` at H267 SHA
`8fdd0012f996524ce8c684b5a5ba698f8a2a9ef8`; H262 original
HTML is checked against SHA256
`3ece76b7f884c96de53a5a4f30d1d24984f01f13b059d4e495ccadd7eb08848a`.
The remote `/kaggle/working` receives ONLY one aggregate-only
JSON: `hala-h268-private-constrained-baseline.json`.

The aggregate reports true model `generate` completion,
whether constrained filter was present DURING generation,
exact H267 raw-output format classification computed in volatile
processor wrapper, coarse click/abstain kind, actual native
dispatch and first-party stock task-state reward, plus explicit
false flags for independent OOD, training admission, fine-tuning,
model improvement, and saved raw media.

All reproducible code/test/docs in the public Hala repo:
`h268/free_t4_constrained_native.py`,
`h268/build_private_payload.py`,
`tests/test_h268_free_t4_constrained_native.py`, and this README.
Only those first-party text sources/tests are public.

On authorized owner Windows checkout with connected Kaggle CLI:

    python -m h268.build_private_payload --source h268/free_t4_constrained_native.py --destination D:\Hala-H268-Private-Upload
    kaggle kernels push -p D:\Hala-H268-Private-Upload --accelerator NvidiaTeslaT4
    kaggle kernels status abdulazizshehri/hala-h268-private-constrained-native-t4

The generated local upload staging directory must be removed once
the private kernel is successfully submitted. Download **ONLY** the
exact output JSON via `--file-pattern
'hala-h268-private-constrained-baseline.json'`. Do not download
all Kaggle outputs or copy pretrained weights to Windows.

## Scientific/ethical gates not waived

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY; H173 scored OOD
(32 cases) is permanently consumed and SEALED; H212 procedural
scored rows barred; H250/H251 independent original app media have
unresolved transitive media licensing, older fork history and decoded
RGBA/PNG leakage. H262 first-party is NOT independent OOD.

Real completion requires legitimately licensed source-disjoint
TRAIN/DEV + new sealed blind source-family OOD, genuinely fitted
LoRA or other new trained weights, paired base-versus-trained native
task-state evaluation, reproducible no-leakage evidence and safe release
qualification. H15 two actual free Colab optimizer steps were an
engineering smoke, NOT a measured native GUI performance improvement.

## Local preflight on owner Windows

The focused H268/H267/H208 source and synthetic tests: **25 passed**.
H268 optimized `python -O`: **4 passed** (standard pytest warning).
Broader H199/H201/H207/H208/H249/H262/H263/H264/H265/H267/H268
regression: **122 passed**, 3019 unrelated deselected, zero failures;
Python compileall passed. These are CPU-only contracts and cannot establish
that the remote model implements the constrained decoding successfully.

## First Kaggle runtime result and narrow compatibility repair

Private Kaggle v1 (correct actual slug
`abdulazizshehri/hala-h268-private-constrained-native-t4`) ended
`ERROR` **before any constrained model generation**, with a verified,
sanitized error class:

`ImportError: cannot import name 'PreTrainedTokenizerBase' from
'transformers.tokenization_utils'`.

The old `lm-format-enforcer==0.11.3` integration imports the tokenizer
base from this legacy module; Transformers 5.16.1 moved the canonical
class to `transformers.tokenization_utils_base`. The package's
misleading secondary error claimed Transformers was not installed,
although H267 had already proved it was installed and ran real Qwen.

H268 now verifies **exact Transformers 5.16.1**, and before importing
the pinned LMFE integration, adds only the SAME canonical class as an
alias in the old module. This compatibility shim:
- never modifies tokenizer contents, logits, output text, model weights,
  strict native JSON checks or coordinates;
- is idempotent and refuses conflicting existing class definitions;
- exists **only in this ephemeral remote research process**;
- has new CPU-only regression coverage.

No neural or GUI result can be claimed for v1. A fresh private v2 run
must show real completed `model.generate` and actual format/task
outcomes before interpretation. The Kaggle metadata ID has also been
corrected to the actual stable slug derived from its title, preventing
later attempts to reference a nonexistent/private wrong ID.
