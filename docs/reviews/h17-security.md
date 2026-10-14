# H17 independent security review

Verdict: no must-fix security finding in the new frozen H17 source. CPU source review only; no GPU execution, model download, adoption, release, merge or paid-compute authorization.

Reviewed source-freeze SHA256: `da58ff694beb8cb8b7da6adbbf08029a96f1890cc3cfe6d7a136d526c42fe5a3`. The ten listed H17 source/config/protocol artifacts matched this externally pinned freeze. The existing H14 manifest and all ten dependency hashes also verified without changes.

## Findings

- Must-fix: none.
- Worth-considering: `evaluation/quantization_runner.py`, model `from_pretrained` call — explicitly add `use_safetensors=True` to make the intended non-pickle weight format a fail-closed contract. The current exact known model revision and `trust_remote_code=False` materially limit exposure; this is hardening, not evidence of a malicious artifact or exploitable fallback at the frozen revision.
- Worth-considering: `evaluation/quantization.py`, `remote_guard` cache check — the root-symlink check does not prove that preexisting cache descendants stay within the ephemeral tree. Before a future GPU run, require a newly created empty cache or validate resolved destinations of existing entries. Hugging Face's ordinary blob/snapshot symlinks can be allowed when their resolved targets remain inside the cache. This concerns a reused or externally altered session cache; fresh-session operation is already required by the protocol.
- Skip-it: adding execution authority, automatic provisioning, paid-session discovery or a claimed security qualification from local d1. None belongs in this experimental source phase.

## Evidence and boundaries

Independent bundled-Python run of `python -m unittest tests.test_quantization -v`: 6/6 passed. Additional `../h17-security-probes.py`: 12 targeted negative checks passed, covering Kaggle, missing Colab marker, all six external-cache variables, root cache symlink and overlength/nonfinite/duplicate-key outputs. Optional `torch`, `transformers`, `bitsandbytes` and `PIL` modules remained unimported. The already independently passed 136-test CPU suite was not repeated here.

Remote/cache/source/media validation precedes optional ML imports. Runtime metadata must match all five exact package pins, including the CUDA wheel suffix. Model and processor loading use the immutable H14 revision with remote code disabled. Quantization settings disable CPU offload; device maps select CUDA device 0, and recorded model/buffer/quantization-state tensors must actually reside on `cuda:0`, including vision storage.

PNG hashes are verified before framework loading and again before each image is decoded; decoded RGB bytes and dimensions must match baseline provenance. Output directories must resolve beneath `/tmp` and cannot preexist. Result persistence uses the unchanged H14-frozen atomic JSON helper. Partial failures retain blocked status and cannot produce a complete paired comparison. Raw text is scored by the unchanged bounded H14 decoder; it is persisted as data and never reaches shell, eval or an action dispatcher. The 48-token generation setting bounds normal output production; parser acceptance is independently limited to 4096 characters.

Environment markers provide an opt-in Colab guard, not independent proof of account billing or a fresh kernel. Free-session selection, environment integrity and cleanup remain the notebook owner's operational prerequisites. No actual optional-framework import, pretrained deserialization, CUDA compatibility, quant-state runtime behavior or GPU residency was exercised by this CPU review.

panel: full — security ✓ fresh-context Codex judge (session model); source review and targeted CPU probes; no local d1 security approval. Other PStack bars belong to the separate combined judge.
