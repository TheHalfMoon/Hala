# H83 — Exact private checkpoint readback identity

**Parent:** H82 Draft PR #66, commit `555b4a41326b720215d2833c98a2730eef6a132d`.
**Status:** CPU-only implementation, provider-free tests. No actual Kaggle upload, no Colab T4 allocation, no trained model checkpoint or evaluation.

## Defect

The existing H65 private checkpoint publisher required private storage, HTTP 403 for anonymous access, and a provider readback flag. However, the H65 caller **did not compare the returned provider reference or provider-readback member SHA256 manifest** against the exact published host bundle. H77 downstream also checked only that the returned checkpoint *step* matched the host; it could accept a purported authenticated readback for another SHA256 checkpoint with the same step. Thus a wrong object or stale/forked checkpoint could be reported as successfully backed up even while the actual host checkpoint remained the only recoverable copy.

## Fix (copy-first, original transport unchanged)

- Keep H61 `publish_bundle` and its authenticated provider readback implementation. Before publication, validate the freshly copied staging bundle via existing `training.checkpoints.load_checkpoint` and ensure its pointer SHA256 and state still match the original host source.
- Compute exact SHA256 for every copied bundle member, member count and byte total. Admit the existing uploader's readback result only when **private/403/readback claims, exact expected provider ref, complete exact file hash map, count and bytes** all match. Reject missing/extra/different provider files and improperly shaped receipts, without deleting the host.
- Re-verify the original host checkpoint and pointer after provider readback; a changed host or source digest cannot silently qualify.
- In H77, require the specific `hala.h65.private-checkpoint-readback.v1` schema and `checkpoint_sha256` equal to the independently parsed host `latest.json` SHA. Recheck unchanged host state and SHA after publication. Record the verified private checkpoint SHA in the H77 receipt without revealing any credential.
- Preserve H77's `FAILED` semantics for partial recovered checkpoints and H65's opt-in host cleanup restriction; no GPU use, provider network calls, paid compute, new model downloads or release.

## Evidence and limitations

Synthetic CPU tests simulate authenticated readback objects and verify **same-step/wrong-SHA refusal**, incorrect provider ref, extra/missing/altered member SHA hashes, wrong count and byte total, mutation while publishing, missing publisher schema, and clean unchanged host copies on failures. Historical positive mocks were updated to mimic the actual publisher response (previously they exposed only booleans). H80 last-chance host salvage + H77 opt-in private-continuation integration remains supported.

These tests prove validation behavior; they **do not** prove that a real provider holds an actual trained checkpoint. An authorized truly free T4 and real optimizer-bearing Qwen step, independent authenticated readback and held-out Arabic/English GUI evaluation remain outstanding.
