"""CPU-only negative/synthetic tests for H426 pinned original tokenizer preflight."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location("h426_pinned_preflight",ROOT/"h426/real_tokenizer_preflight.py")
p=importlib.util.module_from_spec(sp)
sp.loader.exec_module(p)

def row():
    return {
        "image_width":1000,"image_height":500,
        "native_drag_completion":{
            "type":"drag","start_point":[100,50],
            "end_point":[500,250],
            "ordered":True,
            "coordinate_space":"source_image_pixels",
        }
    }

class FakePinnedTokenizer:
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        if tokenize:raise AssertionError("Unexpected template tokenization")
        if add_generation_prompt:return "HEADER:"
        return "HEADER:"+messages[-1]["content"][0]["text"]+"<|im_end|>"
    def __call__(self,text,*,add_special_tokens):
        if add_special_tokens:raise AssertionError("Fake tokenizer must be raw")
        return {"input_ids":[ord(x) for x in text]}
    def decode(self,ids,*,skip_special_tokens):
        if skip_special_tokens:raise AssertionError("Must preserve supervised markers")
        return "".join(chr(i) for i in ids)

class H426RealTokenizerPreflightContract(unittest.TestCase):
    def test_original_pixel_coordinates_to_pinned_1000_json(self):
        answer=p.normalized_source_answer(row())
        data=json.loads(answer)
        self.assertEqual(list(data),["type","start_point","end_point","ordered","coordinate_space"])
        self.assertEqual(data["start_point"],[100,100])
        self.assertEqual(data["end_point"],[500,500])
        self.assertEqual(data["coordinate_space"],"normalized_1000")
        self.assertEqual(answer,separators_canonical(data))

    def test_exact_chat_supervised_tokens_exclude_prefix_and_keep_eot(self):
        value=p.normalized_source_answer(row())
        pieces=p.supervised_pieces(FakePinnedTokenizer(),value)
        self.assertEqual("".join(pieces),value+"<|im_end|>")
        self.assertNotIn("HEADER:","".join(pieces))
        loss_spec=importlib.util.spec_from_file_location("h426_test_loss",ROOT/"h426/target_aware_loss.py")
        loss_mod=importlib.util.module_from_spec(loss_spec)
        loss_spec.loader.exec_module(loss_mod)
        mask=loss_mod.source_coordinate_token_weights(pieces,"slide_resize")
        self.assertEqual(len(mask),len(pieces))
        self.assertEqual(max(mask),16)

    def test_fail_closed_invalid_source_dimensions_and_schema(self):
        bads=[
            {"image_width":0},
            {"image_height":0},
            {"native_drag_completion":{"type":"move"}},
        ]
        for change in bads:
            with self.subTest(change=change),self.assertRaises(ValueError):
                p.normalized_source_answer(dict(row(),**change))

    def test_fail_closed_pixel_coordinates_outside_original_screenshot(self):
        for field,val in (("start_point",[-1,50]),("end_point",[1200,250]),
                          ("start_point",[True,0])):
            r=row()
            r["native_drag_completion"][field]=val
            with self.subTest(field=field,value=val),self.assertRaises(ValueError):
                p.normalized_source_answer(r)

    def test_pinned_training_contract_and_no_model_execution(self):
        self.assertEqual(p.BASE,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(p.REV,"89644892e4d85e24eaac8bacfd4f463576704203")
        self.assertEqual(p.H422_ROWS_SHA,"b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa")
        self.assertEqual(p.QWEN_TOKENIZER_JSON_SHA,"a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7")
        self.assertEqual(p.QWEN_TOKENIZER_CONFIG_SHA,"c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399")
        source=(ROOT/"h426/real_tokenizer_preflight.py").read_text("utf8")
        self.assertIn("H426 original tokenizer masks rejected",source)
        self.assertIn("H426 frozen complete H422 32K source cardinality mismatch",source)
        self.assertNotIn("kaggle kernels push",source)
        self.assertNotIn("model.fit",source)
        self.assertNotIn("torch.cuda",source)
        self.assertIn("local_files_only=True",source)
        self.assertIn("no_private_raw_examples_returned",source)


def separators_canonical(data):
    return json.dumps(data,separators=(",",":"),ensure_ascii=False)

if __name__=="__main__":
    unittest.main()
