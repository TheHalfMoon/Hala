# 0007. Share the representative closed-loop evaluation runtime

Status: accepted (2026-10-16)

## Context

H17 measured three precisions on64 authored fixtures, which cannot decide representative closed-loop quality. H19/H21 already freeze the bilingual simulator, prompts, parser, verifier and FP16 inference callback. Copying that runtime for quantization would introduce divergent task and generation semantics.

## Decision

Add a private optional concrete precision-profile seam to the existing evaluation orchestration. Keep public baseline run defaults, complete QwenInfer AST, SETTINGS and training guard unchanged. A separate H22 entry point requires prospective externally pinned source/config and exact historical bitsandbytes0.50.2/accelerate1.12.0. Reuse H17's quantization configuration and full-vision/quant-state residency check without editing historical files. An external bounded sampler observes cost independently of callback inputs.

## Consequences

The three profiles use exactly the same H19 closed-loop execution and independent scores. H22 remains a distinct profile schema and needs fresh real package/model compatibility, independent source gates and GPU qualification. CPU tests cannot select a precision or adapter. Full-device observations include other processes and sampled maxima do not establish continuous peaks. The narrow optional orchestration branch adds maintenance responsibility without a second task runner.
