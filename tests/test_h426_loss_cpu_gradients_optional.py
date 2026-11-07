"""Optional actual PyTorch CPU autograd: H426 research weight mask, no image/GPU."""
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "h426_grad_loss", ROOT/"h426/target_aware_loss.py")
loss_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_mod)


class FakeCharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            raise ValueError("The actual numeric token text must be retained")
        return "".join(chr(v) for v in ids)


class FakeProcessor:
    tokenizer = FakeCharTokenizer()


def answer():
    return json.dumps({"type":"drag","start_point":[125,350],
                       "end_point":[875,550],"ordered":True,
                       "coordinate_space":"normalized_1000"},
                      separators=(",",":"))


@unittest.skipIf(torch is None, "torch unavailable in source-free CI")
class CPUGradientEvidence(unittest.TestCase):
    def _batch(self, *, hard_end=False):
        answer_string = answer()
        ids = [ord(ch) for ch in answer_string]
        labels = torch.tensor([[-100]+ids], dtype=torch.int64)
        logits = torch.zeros((1, len(ids)+1, 256), dtype=torch.float32)
        with torch.no_grad():
            for i, token in enumerate(ids):
                logits[0, i, token] = 5.0
            if hard_end:
                match = re.search(r'"end_point":\[(\d+),(\d+)\]', answer_string)
                for idx in (*range(match.start(1), match.end(1)),
                            *range(match.start(2),match.end(2))):
                    logits[0, idx, ids[idx]] = -3.0
        return SimpleNamespace(logits=logits.requires_grad_()), labels

    def test_actual_cpu_autograd_nonzero_and_finite(self):
        output, labels = self._batch()
        value = loss_mod.proposed_h426_weighted_ce(
            output, labels, FakeProcessor(), "slide_resize")
        self.assertTrue(torch.isfinite(value).item())
        value.backward()
        self.assertIsNotNone(output.logits.grad)
        self.assertTrue(torch.isfinite(output.logits.grad).all().item())
        self.assertGreater(output.logits.grad.abs().sum().item(), 0)

    def test_endpoint_hard_failure_weight_has_larger_training_loss(self):
        output, labels = self._batch(hard_end=True)
        slider_loss = loss_mod.proposed_h426_weighted_ce(
            output, labels, FakeProcessor(), "slider")
        resize_loss = loss_mod.proposed_h426_weighted_ce(
            output, labels, FakeProcessor(), "slide_resize")
        self.assertGreater(resize_loss.item(), slider_loss.item())

    def test_zero_target_labels_fail_closed(self):
        output, labels = self._batch()
        labels[:] = -100
        with self.assertRaises(ValueError):
            loss_mod.proposed_h426_weighted_ce(output, labels,
                                               FakeProcessor(), "slider")

    def test_multiimage_batch_size_refused(self):
        output, labels = self._batch()
        with self.assertRaises(ValueError):
            loss_mod.proposed_h426_weighted_ce(
                SimpleNamespace(logits=output.logits.repeat(2,1,1)),
                labels.repeat(2,1), FakeProcessor(), "slider")


if __name__=="__main__":
    unittest.main()
