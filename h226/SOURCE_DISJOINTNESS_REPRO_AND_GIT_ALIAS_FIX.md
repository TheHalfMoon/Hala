# H226 — Close source-family clone-suffix bypass in TRAIN/DEV/history

Strict Draft source-only stack: H225 PR #246 exact head `098dc8dbce866ca3e94c53601852e4040a4b129e`.
No merge/release, model weights, training, private historical images or H173 consumed OOD.

## Actual baseline reproduction, before source edits
On owner Windows with exact parent H225, a disposable script with **fabricated metadata only** proved both false accepts:

1. H174 `TRAIN: demo/one` and `DEV: DEMO/ONE.git`, otherwise different app/Git/HTML hashes: `H174_METADATA_ONLY_SPLIT_FIREWALL_PASS_NOT_MODEL_QUALIFIED` (should refuse).
2. Historical owner registry `history/reused` and new TRAIN `HISTORY/REUSED.git`: `H174_SOURCE_HISTORY_CROSSCHECK_ONLY_NOT_DATASET_ADMISSION` (should refuse).

These were failures of a **source-only validation gate**; no owner data, actual source contamination, model score or dataset admission is claimed.

## Fix across gates
- Add `h174.split_firewall.canonical_repository_family()`. Verify existing owner/repo grammar; casefold and recursively remove conventional `.git` clone suffixes, rejecting empty or malformed names. It conservatively treats `owner/repo`, `OWNER/REPO.GIT`, `owner/repo.git.git` as **one family** (including multiple Git clone suffixes).
- Use shared canonical family identity for new split uniqueness in H174; historical registry duplicate detection and candidate-history overlap in `h174.source_history_gate`; and exclusion of the three consumed H137 npm upstream families in H176.
- Keep original raw history records and hashes untouched; no promotion of source candidates. This canonicalization does **not** authenticate upstream provenance.
- `tests/test_h226_clone_repo_family_aliases.py`: source-only positive and adversarial variants for TRAIN/DEV/OOD_RESERVED, consumed H173, historical duplicate aliases, mixed-case or repeated suffix, malformed repos, and retained metadata-only/no-training status.
- Extend the existing H176 negative test to reject repeated `.git.git` aliases of all 3 original H137 npm upstream repos.
- Hosted H226 CPU matrix tests on Windows and Linux, Python 3.11/3.12, zero deps and no model downloads. CI must pass on exact new PR head before it can be called qualified.

## Remaining science/release blockers
H173 consumed OOD is permanently reserved; historical source/fork and old decoded RGBA completeness remain unproven; H212 research sources remain unadmitted and permanently barred from gradients; no TRAIN/DEV/OOD admission, fresh sealed evaluation, Qwen3-VL optimizer step, trained model or release. Past paired 2/32 versus 3/32 evaluation (exact McNemar p=1.0) does not establish improvement.
