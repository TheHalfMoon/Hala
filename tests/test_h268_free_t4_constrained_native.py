"""H268 CPU-only original H199 grammar and token-filter wrapper tests.

Never load Qwen weights or read original private screenshots, historical OOD,
raw model answers, or GPU state in this module.
"""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import json
import re
import unittest

from h199.native_action_sft import parse_actions
from h268.free_t4_constrained_native import (
    X,Y,NATIVE_REGEX,MODEL_ID,MODEL_REVISION,
    LMFE_VERSION,NativeGrammarModel,measure,strict_grammar_proof,
    attest_legacy_tokenizer_alias,
)
from h268.build_private_payload import (
    build_payload,KERNEL_ID,META_NAME,SCRIPT_NAME,
)


class H268ConstrainedNativeTests(unittest.TestCase):
    def test_reference_syntax_and_h199_parser_agree(self):
        strict_grammar_proof()
        good = (0,1,9,10,99,100,699,700,719,899,900,959)
        for x in good:
            for y in (0,1,90,100,699,719):
                if x<=959:
                    val=f'{{"actions":[{{"kind":"click","point":[{x},{y}]}}]}}'
                    self.assertIsNotNone(re.fullmatch(NATIVE_REGEX,val))
                    self.assertEqual(parse_actions(val)["actions"][0]["point"],[x,y])
        self.assertTrue(re.fullmatch(NATIVE_REGEX,'{"actions":[{"kind":"abstain"}]}'))
        for bad in (
            '{"actions":[{"kind":"click","point":[960,0]}]}',
            '{"actions":[{"kind":"click","point":[1,720]}]}',
            '{"actions":[{"kind":"click","point":[001,2]}]}',
            '{"actions":[{"kind":"click","point":[-1,2]}]}',
            '{"actions":[{"kind":"drag","start":[1,2],"end":[3,4]}]}',
            '{"actions":[{"kind":"click","point":[0,0]},{"kind":"abstain"}]}',
            '```json\n{"actions":[{"kind":"abstain"}]}\n```',
            '{}', "null", "I am clicking",
        ):
            with self.subTest(value=bad):
                self.assertIsNone(re.fullmatch(NATIVE_REGEX,bad))
        self.assertEqual(MODEL_ID,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REVISION),40)
        self.assertEqual(LMFE_VERSION,"0.11.3")

    def test_only_canonical_transformers5_tokenizer_class_is_aliased(self):
        class CanonicalTokenizer:
            pass
        class ConflictingTokenizer:
            pass
        legacy=SimpleNamespace()
        self.assertTrue(attest_legacy_tokenizer_alias(legacy,CanonicalTokenizer))
        self.assertIs(legacy.PreTrainedTokenizerBase,CanonicalTokenizer)
        self.assertFalse(attest_legacy_tokenizer_alias(legacy,CanonicalTokenizer))
        conflict=SimpleNamespace(PreTrainedTokenizerBase=ConflictingTokenizer)
        with self.assertRaises(ValueError):
            attest_legacy_tokenizer_alias(conflict,CanonicalTokenizer)
        self.assertIs(conflict.PreTrainedTokenizerBase,ConflictingTokenizer)
        with self.assertRaises(ValueError):
            attest_legacy_tokenizer_alias(SimpleNamespace(),object())

    def test_token_filter_present_during_real_generate_not_posthoc(self):
        model=Mock()
        model.device="cuda:0"
        model.generate.return_value=["opaque tokens"]
        grammar=Mock(return_value=[31])
        counted=NativeGrammarModel(model,grammar)
        self.assertEqual(counted.device,"cuda:0")
        counted.eval()
        out=counted.generate(input_ids=[1,2],max_new_tokens=160)
        self.assertEqual(out,["opaque tokens"])
        self.assertEqual(counted.completed_generations,1)
        self.assertEqual(counted.grammar_installed_calls,1)
        self.assertIs(model.generate.call_args.kwargs["prefix_allowed_tokens_fn"],grammar)
        with self.assertRaises(ValueError):
            counted.generate(prefix_allowed_tokens_fn=lambda *args:[2])
        self.assertEqual(counted.completed_generations,1)
        self.assertEqual(counted.grammar_installed_calls,1)
        model.generate.side_effect=RuntimeError("GPU failure")
        with self.assertRaises(RuntimeError):
            counted.generate(input_ids=[1,2])
        self.assertEqual(counted.completed_generations,1)
        self.assertEqual(counted.grammar_installed_calls,2)

    def test_aggregate_rejects_laundered_accuracy_or_dispatch(self):
        gpu={"real_free_t4":True,"cuda_t4_count":2}
        row=measure(completed=1,filtered=1,
                    format_class="STRICT_NATIVE_JSON",kind="click",
                    dispatched=True,before=3,after=4,gpu=gpu)
        self.assertTrue(row["first_party_task_success"])
        self.assertFalse(row["trained_hala_model_gain"])
        self.assertFalse(row["fresh_source_disjoint_blind_ood"])
        self.assertFalse(row["official_train_dev_ood_admitted"])
        self.assertFalse(row["raw_model_text_screenshots_gold_saved"])
        self.assertFalse(row["kaggle_working_model_cache"])
        self.assertNotIn("coordinates",json.dumps(row))
        for opts in (
            dict(completed=0,filtered=1,format_class="STRICT_NATIVE_JSON",kind="click",dispatched=True),
            dict(completed=1,filtered=1,format_class="MARKDOWN_FENCE",kind="click",dispatched=True),
            dict(completed=1,filtered=0,format_class="STRICT_NATIVE_JSON",kind="click",dispatched=True),
            dict(completed=1,filtered=1,format_class="STRICT_NATIVE_JSON",kind="abstain",dispatched=True),
        ):
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                measure(before=3,after=4,gpu=gpu,**opts)
        row=measure(completed=1,filtered=1,format_class="STRICT_NATIVE_JSON",
                    kind="abstain",dispatched=False,before=3,after=3,gpu=gpu)
        self.assertFalse(row["first_party_task_success"])
        self.assertFalse(row["native_browser_action_dispatched"])

    def test_no_raw_data_or_model_weights_in_private_upload(self):
        root=Path(__file__).resolve().parents[1]
        source=root/"h268"/"free_t4_constrained_native.py"
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)/"fresh"
            receipt=build_payload(source,dest)
            self.assertEqual(receipt["attached_data_or_weights"],0)
            self.assertTrue(receipt["private"])
            names={p.name for p in dest.iterdir()}
            self.assertEqual(names,{META_NAME,SCRIPT_NAME})
            meta=json.loads((dest/META_NAME).read_text("utf-8"))
            code=(dest/SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(meta["id"],KERNEL_ID)
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["enable_gpu"],"true")
            self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"],[])
            self.assertEqual(meta["model_sources"],[])
            self.assertIn("'lm-format-enforcer==0.11.3'",code)
            self.assertIn("install', '--with-deps', 'chromium'",code)
            self.assertIn("HF_HOME=str(root/\"hf\")",code)
            self.assertIn('"/tmp/hala-h268-research"',code)
            self.assertNotIn('HF_HOME="/kaggle/working',code)
            with self.assertRaises(ValueError):
                build_payload(source,dest)


if __name__=="__main__":
    unittest.main()
