# H36 — paired prompt intervention on exposed H34 synthetic DEV

**Status:** Prospective source/protocol only until real Kaggle results are run, independently retained and SHA-replayed. **This experiment is NOT blind TEST, model improvement, release qualification or host action permission.**

## Question and hypothesis

Retained H34 revised-prompt results were fully valid JSON but only **8/32** intended-control hits (3/16 English; 5/16 Arabic), **2/32 wrong visible controls**, and **22/32 points off all controls**. The original H34 prompt gave the concrete sample `{"point":[0.25,0.75]}`; its revised Kaggle raw outputs repeated normalized **y=0.75 in 8/32 attempts, all off observed controls**. The correlation is **post-hoc**, not proof that prompt copying caused the failures.

H36 measures whether removing that concrete coordinate sample from otherwise closely matched instructions alters model output, including the possibility it *reduces* strict parse validity or correct target hits.

## Pre-registered experimental design

- Exactly **32 public, previously exposed H34 synthetic DEV** cases, each with the original SHA-bound PNG, goal and evaluator; 16 English and 16 Arabic, four layouts, four domains. **No new independent examples**.
- **Two arms** on each identical image/goal using the *same* `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203` and FP16 worker running on a permitted Kaggle T4:
  - `numeric_example`: byte-exact historic H34 revised prompt, SHA256 `b257f61644912ae5f2a57b448063579af6477acda614df93338d3782de931555`.
  - `no_numeric_example`: replaces the one concrete-example sentence with text-only schema instruction without sample coordinates, SHA256 `90cf10f977e76fb31a887515645afca7c73d0dc61529c33aaa49b08053b76b5f`.
- **64 inferences**, same VM, one model load, deterministically interleaved pairs. The prompt arm ordered first alternates by case index; both orders occur equally often in both languages. **Plan SHA256:** `39c11abf83cfb997022378d0a3fa516a229ac0209b38de314e8786535671c792`. Frozen H34 case SHA256: `4b7f849a624caac23dd857adb19ed4293d93d0693dbdbd2921d4423b666a6a3a`.
- The H34 strict original `score_attempt` scorer is reused without changing labels, categories, geometry or normalized coordinates. Output includes arm-stratified: **all-attempt denominator 32**, strict JSON validity, correct target, wrong unique target, unique valid binding, abstentions, errors; paired both/only-one/neither correct; English/Arabic stratification.
- **No snapping**, tolerance widening, hidden oracle in prompts, retries that privilege one arm, native device operations, model-training/data changes or retrospective label edits.
- Reproducible new-only private Notebook bundle uses a **clean exact Git commit**, per-file SHA manifest, verified package/RAQM/font, Kaggle Batch-only permission gate, one model and bounded process group with durable per-trial raw journal/partial case recovery.
- **Exact environment pins** inherited from actual H34: torch 2.11.0+cu128, transformers 5.16.1, accelerate 1.14.0, safetensors 0.8.0, Pillow 12.3.0, RAQM 0.10.5, pinned DejaVu license/font, one permitted T4; evaluate actual availability before run.

## Confirmatory readings (DEV intervention ONLY)

Primary: difference in *unconditional* correct target-hit counts across the 32 paired cases, with the 4 paired outcome categories. Secondary: output parse validity, off-target / wrong-target control selections, abstentions, order effects, language and layout metrics.

**Interpretation guard:** Removing a concrete exemplar necessarily changes the schema instruction text; an observed difference cannot uniquely identify a numerical-copy mechanism without further controls. Same exposed 32 DEV cases cannot qualify held-out generalization or Arabic real-UI performance. No claims of improvement, significance or causal mechanism before actual results; even a DEV improvement requires a fresh independently disjoint test and real UI/RTL evaluation.

## Safety and measurement

- Operator may opt in only to **noncommercial**, terms-compliant **zero-cost** free Kaggle GPU without upgrade, payment method or quota circumvention.
- Builder creates the notebook only; it does not submit it or launch a GPU. Requires explicit `--research-allowed` to put the runtime flag in the private notebook.
- Actual worker performs inference only and writes image SHA hashes, protocol/arm binding, source SHA, exact prompts, metadata/partial results. Truncated/changed evidence is rejected on replay. GPU observations are metadata, **not cryptographic attestation**.
- Replay requires an independently obtained SHA256 of `raw_attempts.jsonl`, complete 32-image manifest with PNG CRC/format verification, exact case/arm sequence, prompt hashes and H34 strict score agreement. Retain all raw evidence in GitHub under a new H36 evidence directory if run.
- Local synthetic fixture tests are **not model measurements**.

### Prepare privately (after signed reviewed source commit)

```powershell
python -m unittest tests.test_h36_paired_prompt -v
python -m unittest discover -s tests
python -m tools.build_h36_kaggle --root . --source-commit <verified-source-sha> --output <new-local-bundle-folder> --research-allowed
```

Use a **clean source worktree**; do not delete or commit another agent's untracked `.ignore`. Verify the private kernel metadata (GPU:true; no billing), then manually/explicitly authorize any Kaggle API submission. Run only if the available GPU/quota/terms are compatible; do not launch if any required pin differs. After retrieving all output files, compute externally `sha256(raw_attempts.jsonl)` and run:

```powershell
python -m h36.audit --output <retrieved-evidence-dir> --expected-raw-sha256 <digest>
```

Publish raw evidence and diagnostics immutably after verification. If GPU is unavailable, keep the source-only PR as **unexecuted** and continue independent engineering.

## Verified local source gates — 2026-10-20

On Windows Abdulaziz: `python -m unittest tests.test_h36_paired_prompt -v` **7 passed**, full Python suite `python -m unittest discover -s tests` **494 run / 492 passed / 2 skipped**, `compileall` PASS. Genuine Graft **wiring-only** build/check PASS **1,543 nodes / 4,079 edges**, with semantic meaning tier 0% (not run). No PStack fresh-context review, Alibaba LLM code review, D1 review or actual Kaggle inference has yet occurred for H36. Actual authorized Kaggle quota observed read-only: **29.03h remaining / 30.00h total GPU**, refresh 2026-10-24; still subject to per-run hardware and terms admission. Successful unit tests are source/fixture tests, not visual model evaluation.

## Open gates

No new results yet; no cross-domain unseen TEST, model correction, native actions, Arabic expert adjudication, real screenshot evaluation, H23 actual CUDA cancellation/worker restart or founder production release authorization. The goal remains to build a **genuinely working Hala**, not to substitute project documentation for model quality.
