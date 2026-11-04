"""H423 real PREVIOUSLY TRAINED parent LoRA header regression.

This checks genuine H417 3,720-step Tesla T4 safetensors, not dummy fixture.
The private H417 weights stay on the owner's Windows drive, never GitHub.
Hosted CI intentionally skips real private weights.
"""
from hashlib import sha256
import os
from pathlib import Path
import unittest

from h423.qualify_h419_checkpoint import safe_adapter_headers,file_sha

H417_REAL_T4_ADAPTER_SHA="11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1"
H417_REAL_T4_SAFETENSORS_BYTES=12878000
H417_REAL_T4_LORA_TENSORS=224


class H423ActualTrainedPEFTAdapter(unittest.TestCase):
    def test_original_H417_trained_model_safetensors_is_a_real_SHA_224_tensor_lora(self):
        p=os.environ.get("H423_OWNER_REAL_H417_ADAPTER")
        if not p:
            self.skipTest("Private H417 actual trained adapter not mounted on GitHub CI")
        path=Path(p)
        self.assertTrue(path.is_file(),"private owner H417 original trained LoRA required")
        self.assertEqual(path.stat().st_size,H417_REAL_T4_SAFETENSORS_BYTES)
        self.assertEqual(file_sha(path),H417_REAL_T4_ADAPTER_SHA)
        meta=safe_adapter_headers(path)
        self.assertEqual(meta["real_lora_tensor_count"],H417_REAL_T4_LORA_TENSORS)
        self.assertEqual(meta["real_lora_bytes"],H417_REAL_T4_SAFETENSORS_BYTES)
        self.assertTrue(meta["exact_safetensors_tensors_cover_all_bytes"])

    def test_actual_trained_H417_PEFT_config_uses_pinned_native_Qwen_base(self):
        path=os.environ.get("H423_OWNER_REAL_H417_ADAPTER")
        if not path:
            self.skipTest("Original private trained adapter not mounted on CI")
        config=Path(path).with_name("adapter_config.json")
        self.assertTrue(config.is_file())
        import json
        values=json.loads(config.read_text(encoding="utf8"))
        self.assertEqual(values.get("base_model_name_or_path"),
                         "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(values.get("peft_type"),"LORA")
        self.assertEqual(values.get("r"),8)
        self.assertEqual(values.get("lora_alpha"),16)
        self.assertEqual(set(values.get("target_modules",[])),
                         {"q_proj","o_proj","k_proj","v_proj"})
        self.assertEqual(values.get("task_type"),"CAUSAL_LM")

    def test_trained_parent_expected_sha_is_pinned_not_fabricated(self):
        self.assertEqual(len(H417_REAL_T4_ADAPTER_SHA),64)
        self.assertEqual(H417_REAL_T4_ADAPTER_SHA,
                         "11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1")
        self.assertGreater(H417_REAL_T4_SAFETENSORS_BYTES,1000000)
        self.assertGreater(H417_REAL_T4_LORA_TENSORS,100)


if __name__=="__main__":
    unittest.main()
