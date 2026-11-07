"""H425/H426 objective proof on invented CPU drag tokens.

No Qwen weights, screenshot pixels, source dataset, GPU, optimizer or network.
The result is mathematical objective evidence, NEVER learned GUI accuracy.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


def run() -> dict:
    import torch
    from h426.h425_loss_bridge import compare_original_h425_to_h426
    if torch.cuda.is_initialized():
        raise RuntimeError("H426 probe must not initialize GPU")
    torch.set_num_threads(1)

    class CharTokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            if skip_special_tokens:
                raise ValueError("H426 probe forbids removing native target tokens")
            return "".join(chr(v) for v in ids)

    processor = SimpleNamespace(tokenizer=CharTokenizer())
    answer = json.dumps(
        {"type": "drag", "start_point": [100, 250],
         "end_point": [850, 520], "ordered": True,
         "coordinate_space": "normalized_1000"},
        separators=(",", ":"),
    )
    tokens = [ord(c) for c in answer]
    labels = torch.tensor([[-100] * 8 + tokens], dtype=torch.long)
    vulnerable = answer.index('"end_point":[') + len('"end_point":[')
    damaged_offsets = list(range(vulnerable, vulnerable + 3))
    damaged_offsets += list(range(vulnerable + 4, vulnerable + 7))
    results = {}
    for domain in ("slider", "slide_resize", "text_highlight"):
        logits = torch.zeros((1, labels.shape[1], 256), dtype=torch.float32)
        with torch.no_grad():
            for offset, token in enumerate(tokens):
                logits[0, 7 + offset, token] = 5.0
            for offset in damaged_offsets:
                logits[0, 7 + offset, tokens[offset]] = -3.0
        logits.requires_grad_()
        source = {
            "id": "h426-invented-drag-cpu-1",
            "image_sha256": "a" * 64,
            "domain": domain,
            "language": "en",
            "body": "Move the synthetic rectangular object to a new position.",
        }
        compared = compare_original_h425_to_h426(
            SimpleNamespace(logits=logits), labels, processor, source,
        )
        old, new = compared["h425_original_loss"], compared["h426_target_loss"]
        new.backward()
        gradient = logits.grad
        if (not bool(torch.isfinite(old)) or not bool(torch.isfinite(new))
            or gradient is None or not bool(torch.isfinite(gradient).all())
            or not bool((gradient.abs() > 0).any())):
            raise ValueError("H426 CPU probe refused nonfinite or absent gradient")
        results[domain] = {
            "h425_source_baseline_loss": round(old.item(), 6),
            "h426_domain_target_loss": round(new.item(), 6),
            "gradient_abs_sum": round(gradient.abs().sum().item(), 6),
        }
    if not (results["slide_resize"]["h426_domain_target_loss"]
            > results["text_highlight"]["h426_domain_target_loss"]
            > results["slider"]["h426_domain_target_loss"]):
        raise ValueError("H426 domain-target objective ordering drift")
    return {
        "schema": "hala.h426.synthetic-original-h425-objective-cpu.v1",
        "torch_cpu_version": torch.__version__,
        "source_images_or_model_weights_read": False,
        "train_steps": 0,
        "independent_ood_accuracy_measured": False,
        "rights_review_completed": False,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
