# H10 paired quantization and residency qualification

Depends on: qualified unchanged multimodal checkpoint and real H3 provider. Surface: experiments, model_cards, inference adapters.

Acceptance: exact base/processor/tokenizer/vision files; one supported precision at a time; BF16/FP16 vs 8-bit vs4-bit on identical heldout cases, same prompts/frame geometry/budgets; real vision processing and text generation; disk bytes, loaded/peak VRAM, cold/warm latency, Arabic/English, schema/wrong-target/task success and uncertainty. Model-specific vLLM/SGLang/Transformers compatibility must be tested rather than inferred.

Hardstop: no unsupported forced quantization, active-MoE/resident-weight confusion or unpaired “equivalent accuracy” claim. No local multi-GB weights or paid resources.
