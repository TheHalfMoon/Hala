"""H61 integration test of the trainer's H61 curriculum path.

The H61 training path touches four things that only meet on the real session:
the H61 curriculum loader, the completion label boundary, the coordinate
supervision weights and the weighted loss. A defect in any of them would
only be discovered after a free GPU session has been waiting for hours, so
this test exercises them together with a deterministic mock processor.

The real tokenizer and the real curriculum rows are proved separately by
`tools/preflight_h61_processor.py`, whose receipt is in
`reviews/h61-processor-preflight-20261021/`.
"""
import json
import tempfile
import unittest
from pathlib import Path

try:  # The CPU qualification environment has no torch.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

from training.masking import completion_labels
from training.scripts.train_adapter import prepare_inputs

from h61.curriculum import export as export_curriculum

SENTINEL = "\n<|endoftext|>\n"


class MockProcessor:
    """A deterministic whitespace tokenizer with the Qwen chat template shape."""

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
                continue
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"])
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<|assistant|>\n"
        else:
            text += SENTINEL
        return text

    def __call__(self, text=None, images=None, return_tensors=None):
        rows = text if isinstance(text, list) else [text]
        encoded = []
        for row in rows:
            encoded.append([hash_token(word) for word in row.split()])
        if return_tensors == "pt":
            import torch as _torch

            width = max(len(row) for row in encoded)
            padded = [row + [0] * (width - len(row)) for row in encoded]
            return _torch.tensor(padded, dtype=_torch.long)
        return {"input_ids": encoded}

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(TOKEN_TEXT.get(int(token), "") for token in ids)


TOKEN_TEXT = {}


def hash_token(word):
    if word not in TOKEN_TEXT:
        TOKEN_TEXT[word] = 1000 + len(TOKEN_TEXT)
    return TOKEN_TEXT[word]


def make_shard(root: Path, domain: str = "slider", images: int = 4, rows: int = 1):
    shard = root / f"shard-{domain}"
    shard.mkdir(parents=True, exist_ok=True)
    labels = []
    for index in range(images):
        image_id = f"{domain}-{index:04d}"
        for task_index in range(rows):
            labels.append({
                "id": f"h60-{domain}-{image_id}-t{task_index:03d}",
                "split": "train", "domain": domain, "subtype": None, "language": "en",
                "image_id": image_id, "image_path": f"images/{domain}/{image_id}.jpg",
                "image_sha256": "0" * 64, "image_width": 1024, "image_height": 640,
                "source_task_json_path": f"tasks/{domain}/{image_id}.json",
                "source_task_sha256": "1" * 64, "source_task_index": task_index,
                "source_image_tar_offset": 0, "source_task_tar_offset": 0,
                "intent": "Set the brightness to 6", "action_type": "drag", "ordered": True,
                "start_bbox": [591, 246, 601, 256], "end_bbox": [711, 246, 721, 256],
                "completion": {
                    "type": "drag", "start_point": [596, 251], "end_point": [716, 251],
                    "coordinate_space": "source_image_pixels", "ordered": True,
                },
            })
    blob = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in labels
    ).encode("utf8")
    (shard / "labels.jsonl").write_bytes(blob)
    (shard / "manifest.json").write_text(json.dumps({
        "schema": "hala.h60.dragon-train-shard.v1",
        "shard_id": f"h60-dragon-train-{domain}-0000",
        "source_domain": domain, "split": "train",
        "source_commit_sha": "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
        "upstream_full_tar_lfs_sha256": "2" * 64,
        "official_eval_or_test_used": False, "benchmark_answer_leakage": False,
        "original_images": images, "native_drag_actions": len(labels),
        "labels_sha256": hashlib_sha256(blob),
    }), encoding="utf8")
    return shard


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class TrainerH61PathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="h61-trainer-"))

    def test_the_completion_boundary_masks_the_prompt(self):
        prefix = [1, 2, 3]
        full = [1, 2, 3, 4, 5, 6]
        labels = completion_labels(full, prefix, 64)
        self.assertEqual(labels, [-100, -100, -100, 4, 5, 6])

    def test_the_completion_boundary_refuses_a_broken_prefix(self):
        with self.assertRaises(ValueError):
            completion_labels([1, 2, 3], [9, 9], 64)
        with self.assertRaises(ValueError):
            completion_labels([1, 2, 3], [1, 2, 3], 2)

    @unittest.skipUnless(torch is not None, "torch is required for the tensor path")
    def test_prepare_inputs_marks_the_coordinate_tokens(self):
        shard = make_shard(self.root)
        export_curriculum(self.root, target_rows=2, max_rows_per_image=1)
        rows = [json.loads(line) for line in
                (self.root / "h61_drag_curriculum.jsonl").read_text("utf8").splitlines() if line]
        row = rows[0]
        image = self.root / row["image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (1024, 640), (10, 20, 30)).save(image, format="PNG")
        row["image_sha256"] = hashlib_sha256(image.read_bytes())
        inputs, labels, weights = prepare_inputs(
            row, MockProcessor(), self.root, 4096, native_drag_jpeg=False,
            supervision="coordinate_weighted", coordinate_weight=8.0, return_weights=True,
        )
        supervised = (labels[0] != -100).nonzero().flatten().tolist()
        self.assertTrue(supervised)
        heavy = (weights[0] > 1.0).nonzero().flatten().tolist()
        self.assertTrue(heavy)
        self.assertTrue(set(heavy) <= set(supervised))
        self.assertTrue(all(float(weights[0][index]) == 8.0 for index in heavy))
        self.assertTrue(all(float(weights[0][index]) == 1.0
                            for index in supervised if index not in set(heavy)))

    @unittest.skipUnless(torch is not None, "torch is required for the tensor path")
    def test_the_weighted_loss_equals_the_hand_computed_weighted_cross_entropy(self):
        from h61.supervision import weighted_lm_loss

        torch.manual_seed(3)
        logits = torch.randn(1, 6, 13)
        labels = torch.tensor([[-100, 2, 3, 4, 5, 6]])
        weights = torch.tensor([[0.0, 1.0, 8.0, 8.0, 1.0, 1.0]])
        loss = weighted_lm_loss(logits, labels, weights)
        plain = torch.nn.functional.cross_entropy(
            logits[0, :-1, :].float(), labels[0, 1:], reduction="none"
        )
        manual = (plain * weights[0, 1:]).sum() / weights[0, 1:].sum()
        self.assertTrue(torch.allclose(loss, manual, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
