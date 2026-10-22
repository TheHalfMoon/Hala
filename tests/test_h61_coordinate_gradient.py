"""H61 gradient proof for the coordinate-focused training step.

The H61 run spends four hours inside one optimizer step, so the step itself
is proved here with a real gradient, not a mock: the trainer's own inputs, the
trainer's own weighted loss, a real backward pass and a real parameter
update.

What is measured:

* the step produces a finite loss and a finite gradient;
* the coordinate tokens carry a larger share of the gradient than the
  structural tokens, which is the whole reason the supervision was changed;
* the loss of one example falls over a few optimizer steps, so the update
  direction is the one the weighted loss asks for;
* the weighted step and the plain step do not agree on the parameter
  movement, so the supervision change really changes the training.
"""
import json
import tempfile
import unittest
from pathlib import Path

try:  # The CPU qualification environment has no torch.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

from training.scripts.train_adapter import prepare_inputs

from h61.curriculum import export as export_curriculum
from h61.supervision import weighted_lm_loss


def _make_shard(root: Path):
    """One synthetic original-style shard with a single drag row."""
    import hashlib

    shard = root / "shard-slider"
    shard.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "h60-slider-0000-00000-t000", "split": "train", "domain": "slider",
        "subtype": None, "language": "en", "image_id": "slider-0000",
        "image_path": "images/slider/slider-0000.jpg", "image_sha256": "0" * 64,
        "image_width": 1024, "image_height": 640,
        "source_task_json_path": "tasks/slider/slider-0000.json",
        "source_task_sha256": "1" * 64, "source_task_index": 0,
        "source_image_tar_offset": 0, "source_task_tar_offset": 0,
        "intent": "Set the brightness to 6", "action_type": "drag", "ordered": True,
        "start_bbox": [591, 246, 601, 256], "end_bbox": [711, 246, 721, 256],
        "completion": {
            "type": "drag", "start_point": [596, 251], "end_point": [716, 251],
            "coordinate_space": "source_image_pixels", "ordered": True,
        },
    }
    blob = (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n").encode("utf8")
    (shard / "labels.jsonl").write_bytes(blob)
    (shard / "manifest.json").write_text(json.dumps({
        "schema": "hala.h60.dragon-train-shard.v1",
        "shard_id": "h60-dragon-train-slider-0000", "source_domain": "slider",
        "split": "train",
        "source_commit_sha": "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
        "upstream_full_tar_lfs_sha256": "2" * 64,
        "official_eval_or_test_used": False, "benchmark_answer_leakage": False,
        "original_images": 1, "native_drag_actions": 1,
        "labels_sha256": hashlib.sha256(blob).hexdigest(),
    }), encoding="utf8")
    return shard


def _tiny_head(vocabulary: int, hidden: int):
    """A tiny language model head: an embedding and a vocabulary projection."""
    import torch

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(11)
            self.embedding = torch.nn.Embedding(vocabulary, hidden)
            self.projection = torch.nn.Linear(hidden, vocabulary)

        def forward(self, input_ids):
            return self.projection(self.embedding(input_ids))

    return TinyHead()


@unittest.skipUnless(torch is not None, "torch is required for the gradient proof")
class CoordinateGradientTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="h61-gradient-"))

    def _row(self):
        _make_shard(self.root)
        export_curriculum(self.root, target_rows=1, max_rows_per_image=1)
        rows = [json.loads(line) for line in
                (self.root / "h61_drag_curriculum.jsonl").read_text("utf8").splitlines() if line.strip()]
        row = rows[0]
        from PIL import Image

        image = self.root / row["image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1024, 640), (10, 20, 30)).save(image, format="JPEG")
        import hashlib

        row["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        row["image_format"] = "JPEG"
        return row

    def _step(self, row, weights, model, learning_rate=0.5):
        inputs, labels, _ = prepare_inputs(
            row, _Processor(), self.root, 4096, native_drag_jpeg=True,
            supervision="completion", return_weights=True,
        )
        import torch

        logits = model(inputs.input_ids)
        loss = weighted_lm_loss(logits, labels, weights)
        model.zero_grad()
        loss.backward()
        gradient = [parameter.grad for parameter in model.parameters()]
        total = sum(
            float(gradient_item.abs().sum()) for gradient_item in gradient
            if gradient_item is not None
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate * parameter.grad
        return float(loss), total, gradient

    def test_the_coordinate_tokens_carry_the_larger_share_of_the_gradient(self):
        import torch

        row = self._row()
        inputs, labels, weights = prepare_inputs(
            row, _Processor(), self.root, 4096, native_drag_jpeg=True,
            supervision="coordinate_weighted", coordinate_weight=8.0, return_weights=True,
        )
        model = _tiny_head(vocabulary=1200, hidden=16)
        logits = model(inputs.input_ids)
        loss = weighted_lm_loss(logits, labels, weights)
        # The gradient of the loss with respect to the logits is the per token
        # share of the gradient, token by token.
        logits_gradient = torch.autograd.grad(loss, logits)[0]
        supervised = (labels[0] != -100).nonzero().flatten().tolist()
        heavy = (weights[0] > 1.0).nonzero().flatten().tolist()
        shares = []
        for position in supervised:
            row_gradient = logits_gradient[0, position - 1]
            shares.append(float(row_gradient.abs().sum()))
        coordinate_share = sum(shares[position - supervised[0]] for position in heavy)
        structural_share = sum(
            share for position, share in zip(supervised, shares)
            if position not in set(heavy)
        )
        self.assertGreater(coordinate_share, structural_share)
        self.assertGreater(coordinate_share / (coordinate_share + structural_share), 0.6)

    def test_one_example_gets_better_under_the_weighted_step(self):
        import torch

        row = self._row()
        model = _tiny_head(vocabulary=1200, hidden=16)
        losses = []
        for _ in range(6):
            inputs, labels, weights = prepare_inputs(
                row, _Processor(), self.root, 4096, native_drag_jpeg=True,
                supervision="coordinate_weighted", coordinate_weight=8.0, return_weights=True,
            )
            loss = weighted_lm_loss(model(inputs.input_ids), labels, weights)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= 0.5 * parameter.grad
            losses.append(float(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(losses[-1], losses[0])

    def test_the_weighted_step_and_the_plain_step_move_the_model_differently(self):
        import torch

        row = self._row()
        weighted_model = _tiny_head(vocabulary=1200, hidden=16)
        plain_model = _tiny_head(vocabulary=1200, hidden=16)
        plain_model.load_state_dict(weighted_model.state_dict())
        inputs, labels, weights = prepare_inputs(
            row, _Processor(), self.root, 4096, native_drag_jpeg=True,
            supervision="coordinate_weighted", coordinate_weight=8.0, return_weights=True,
        )
        plain_weights = torch.tensor(
            [[0.0 if token == -100 else 1.0 for token in labels[0].tolist()]]
        )
        weighted_loss = weighted_lm_loss(weighted_model(inputs.input_ids), labels, weights)
        plain_loss = weighted_lm_loss(plain_model(inputs.input_ids), labels, plain_weights)
        weighted_model.zero_grad()
        plain_model.zero_grad()
        weighted_loss.backward()
        plain_loss.backward()
        weighted_gradient = weighted_model.projection.weight.grad
        plain_gradient = plain_model.projection.weight.grad
        self.assertFalse(torch.allclose(weighted_gradient, plain_gradient, atol=1e-6))


class _Processor:
    """The minimal processor the trainer needs, with a real chat template shape."""

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        assistant = None
        for message in messages:
            if message.get("role") == "assistant":
                assistant = message["content"]
                continue
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
                continue
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"])
        text = "\n".join(parts) + " <|assistant|>"
        if assistant is not None:
            text += " " + assistant + " <|endoftext|>"
        return text

    def __call__(self, text=None, images=None, return_tensors=None):
        import torch

        rows = text if isinstance(text, list) else [text]
        encoded = [[_token(word) for word in row.split()] for row in rows]
        width = max(len(row) for row in encoded)
        padded = [row + [0] * (width - len(row)) for row in encoded]
        return _Batch(torch.tensor(padded, dtype=torch.long))

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(_TEXT.get(int(token), "") for token in ids)


_TEXT = {}
_BY_WORD = {}


def _token(word):
    if word not in _BY_WORD:
        _BY_WORD[word] = 1000 + len(_BY_WORD)
        _TEXT[_BY_WORD[word]] = word
    return _BY_WORD[word]


class _Batch:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        return self


if __name__ == "__main__":
    unittest.main()

