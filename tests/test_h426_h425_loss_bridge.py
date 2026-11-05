"""H426 meets REAL H425 loss source; synthetic-only research contract.

No original source screenshots, trained model, GPU, or optimizer operations.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "h426_bridge_test", ROOT/"h426/h425_loss_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def case(domain="slide_resize"):
    return {
        "id":"synthetic-source-case-1",
        "image_sha256":"a"*64,
        "domain":domain,
        "language":"en",
        "body":"Drag synthetic shape from top left to bottom right.",
    }


class H426IdentityAndBoundary(unittest.TestCase):
    def test_real_h425_shape_and_provenance_are_required(self):
        x=bridge.source_case_identity(case())
        self.assertEqual(x["domain"],"slide_resize")
        self.assertEqual(len(x["source_identity_sha"]),64)
        self.assertNotIn("id",x)
        self.assertNotIn("body",x)

    def test_reject_non_case_source_identity(self):
        bad=[
            None,
            {"domain":"slide_resize"},
            dict(case(),domain="injected"),
            dict(case(),language="fr"),
            dict(case(),image_sha256="AA"*32),
            dict(case(),image_sha256="0"*63),
            dict(case(),id=""),
            dict(case(),body=""),
            dict(case(),language=True),
        ]
        for row in bad:
            with self.subTest(row=str(row)[:80]), self.assertRaises(ValueError):
                bridge.source_case_identity(row)

    def test_cpu_only_original_numeric_vs_target_weights_are_distinct(self):
        original=json.dumps({
            "type":"drag","start_point":[100,250],
            "end_point":[850,520],"ordered":True,
            "coordinate_space":"normalized_1000"
        },separators=(",",":"))
        pieces=list(original)+["<|im_end|>"]
        expected={
            "slide_resize":(6,6,0),
            "text_highlight":(0,0,12),
            "slider":(12,0,0),
        }
        for domain,(eights,sixteens,twelves) in expected.items():
            with self.subTest(domain=domain):
                out=bridge.original_vs_target_weight_diagnostic(pieces,domain)
                self.assertEqual(out["tokens"],len(pieces))
                self.assertEqual(out["h425_numeric_weight_8_tokens"],16)
                self.assertEqual(out["h426_weight_8_tokens"],eights)
                self.assertEqual(out["h426_weight_12_tokens"],twelves)
                self.assertEqual(out["h426_weight_16_tokens"],sixteens)
                self.assertNotEqual(out["h426_weight_sum"],out["h425_weight_sum"])
                self.assertFalse(out["model_inference_or_training"])
                self.assertFalse(out["original_private_data_included"])

    def test_cpu_weight_diagnostic_refuses_an_unparseable_source_target(self):
        for pieces,dom in ((["garbage"],"slider"),
                           ([], "slider"),
                           (["valid"],"unknown")):
            with self.subTest(domain=dom), self.assertRaises(ValueError):
                bridge.original_vs_target_weight_diagnostic(pieces,dom)

    def test_no_backend_launch_or_model_publication(self):
        text=(ROOT/"h426/h425_loss_bridge.py").read_text("utf8")
        self.assertIn("h425_h417_replay_paired.py",text)
        self.assertIn("h426/target_aware_loss.py",text)
        self.assertNotIn("torch.optim",text)
        self.assertNotIn("optimizer.step(",text)
        self.assertNotIn("kaggle kernels push",text)
        self.assertNotIn("model.save_pretrained(",text)


class SyntheticTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("H426 must retain original field marker characters")
        return "".join(chr(i) for i in ids)


class SyntheticProcessor:
    tokenizer = SyntheticTokenizer()


def make_batch(*,hard_endpoint=False):
    raw=json.dumps({
        "type":"drag","start_point":[100,250],"end_point":[850,520],
        "ordered":True,"coordinate_space":"normalized_1000"
    },separators=(",",":"))
    answer=[ord(ch) for ch in raw]
    prefix=[-100]*8
    label=torch.tensor([prefix+answer],dtype=torch.long)
    # Out-of-the-box H425 tensor shape: [batch=1, header+assistant, vocab].
    logits=torch.zeros((1,len(prefix)+len(answer),256),dtype=torch.float32)
    with torch.no_grad():
        for idx,v in enumerate(answer):
            logits[0,8+idx-1,v]=5.0
        if hard_endpoint:
            anchor=raw.index('"end_point":[')+len('"end_point":[')
            positions=list(range(anchor,anchor+3))+list(range(anchor+4,anchor+7))
            for idx in positions:
                logits[0,8+idx-1,answer[idx]]=-3
    return SimpleNamespace(logits=logits.requires_grad_()),label


@unittest.skipIf(torch is None,"Torch unavailable on source-only GitHub CPU CI")
class TorchRealH425BridgeEvidence(unittest.TestCase):
    def test_real_h425_source_criterion_and_h426_on_identical_logits(self):
        model,labels=make_batch(hard_endpoint=True)
        out=bridge.compare_original_h425_to_h426(
            model,labels,SyntheticProcessor(),case("slide_resize"))
        self.assertEqual(out["domain"],"slide_resize")
        self.assertTrue(out["no_training_performed"])
        self.assertTrue(torch.isfinite(out["h425_original_loss"]).item())
        self.assertTrue(torch.isfinite(out["h426_target_loss"]).item())
        self.assertGreater(out["h426_target_loss"].item(),
                           out["h425_original_loss"].item())
        out["h426_target_loss"].backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all().item())
        self.assertGreater(model.logits.grad.abs().sum().item(),0)

    def test_actual_h425_and_h426_losses_differ_on_same_slider(self):
        # Force unequal token-wise CE; on an all-equal CE tensor, every
        # normalized positive weighting must have the same mean.
        model,labels=make_batch(hard_endpoint=True)
        out=bridge.compare_original_h425_to_h426(
            model,labels,SyntheticProcessor(),case("slider"))
        self.assertNotEqual(out["h425_original_loss"].item(),
                            out["h426_target_loss"].item())

    def test_refuse_fragmented_assistant_target(self):
        model,labels=make_batch()
        labels[0,-4]=-100
        with self.assertRaisesRegex(ValueError,"suffix fragmented"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())

    def test_refuse_out_of_vocabulary_label(self):
        model,labels=make_batch()
        labels[0,-1]=257
        with self.assertRaisesRegex(ValueError,"malformed labels"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())

    def test_refuse_unmasked_source_prompt_even_if_assistant_is_valid(self):
        model,labels=make_batch()
        labels[0,0]=77
        with self.assertRaisesRegex(ValueError,"leaked prompt"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())

    def test_refuse_negative_token_id_outside_ignore_mask(self):
        model,labels=make_batch()
        labels[0,-1]=-1
        with self.assertRaisesRegex(ValueError,"malformed labels"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())

    def test_refuse_bad_logit_alignment(self):
        model,labels=make_batch()
        model.logits=model.logits[:,:-1,:]
        with self.assertRaisesRegex(ValueError,"tensor alignment drift"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())

    def test_refuse_nonfinite_supervised_logits(self):
        model,labels=make_batch()
        with torch.no_grad():
            model.logits[0,-2,20]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            bridge.compare_original_h425_to_h426(
                model,labels,SyntheticProcessor(),case())


if __name__=="__main__":
    unittest.main()
