"""Synthetic H199 action/SFT schema, completion-mask and release firewall."""
import json
import math
import unittest

from h199.native_action_sft import (
    ADMITTED_TRAIN_MANIFEST_SHAS, WIDTH, HEIGHT,
    assistant_completion_mask, parse_actions, source_task_prompt,
    synthetic_cpu_sft_dry_run, training_messages_for_synthetic_contract,
    verify_real_training_release,
)


def payload(action):
    return json.dumps({"actions": [action]}, ensure_ascii=False)


def release(**changes):
    v = {
        "schema": "hala.original-native-train-release.v1",
        "manifest_sha256": "a"*64,
        "split": "TRAIN",
        "official_admitted": True,
        "complete_source_history_proven": True,
        "complete_old_rgba_proven": True,
        "sealed_dev_gold": True,
        "consumed_ood_reused": False,
    }
    v.update(changes)
    return v


class H199NativeSFTTests(unittest.TestCase):
    def test_three_original_gui_native_actions_and_separate_abstain(self):
        self.assertEqual(parse_actions(payload({
            "kind":"click","point":[0,0]
        }))["actions"][0]["kind"],"click")
        self.assertEqual(parse_actions(payload({
            "kind":"drag","start":[0,0],"end":[959.99,719.5]
        }))["actions"][0]["kind"],"drag")
        self.assertEqual(parse_actions(payload({
            "kind":"key_press","point":[100,200],"key":"ArrowRight"
        }))["actions"][0]["key"],"ArrowRight")
        self.assertEqual(parse_actions(payload({"kind":"abstain"}))["actions"][0]["kind"],"abstain")

    def test_no_invalid_hidden_dom_or_extra_action_fields(self):
        for item in [
            {"kind":"click","point":[1,2],"selector":"#secret"},
            {"kind":"key_press","point":[20,30],"key":"Enter","dom_id":"a"},
            {"kind":"drag","start":[1,2],"end":[2,3],"source_label":"gold"},
            {"kind":"abstain","reason":"model cannot use oracle"},
        ]:
            with self.subTest(item=item),self.assertRaises(ValueError):
                parse_actions(payload(item))

    def test_reject_invalid_keyboard_shortcuts_arbitrary_text_or_shell(self):
        for key in ("Ctrl+Alt+Delete","Ctrl+V","cmd","rm -rf /",
                    "ArrowRight\nEnter","A","Shift+Tab",""):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"keyboard"):
                parse_actions(payload({"kind":"key_press","point":[5,9],"key":key}))

    def test_bool_nan_inf_negative_and_outside_pixels_denied(self):
        for x,y in ((True,10),(float("nan"),4),(float("inf"),3),
                    (-1,2),(960,1),(1,720),("23",44),([1],2)):
            with self.subTest(value=(x,y)),self.assertRaises(ValueError):
                parse_actions(payload({"kind":"click","point":[x,y]}))
        self.assertEqual((WIDTH,HEIGHT),(960,720))

    def test_zero_distance_drag_rejected(self):
        with self.assertRaisesRegex(ValueError,"zero-distance"):
            parse_actions(payload({
                "kind":"drag","start":[300,200],"end":[300,200]
            }))

    def test_exact_single_action_only_and_bounded_json(self):
        for raw in (
            '{"actions":[]}', '{"actions":[{"kind":"abstain"},{"kind":"abstain"}]}',
            '{"action":{"kind":"abstain"}}',
            '{"actions":[{"kind":"abstain"}],"gold":true}',
            '{"actions":[', "a"*1300, '{"actions":null}',
        ):
            with self.subTest(raw=raw[:45]),self.assertRaises(ValueError):
                parse_actions(raw)

    def test_abstain_not_valid_supervision(self):
        with self.assertRaisesRegex(ValueError,"abstain"):
            parse_actions(payload({"kind":"abstain"}),training_target=True)

    def test_synthetic_bilingual_sft_has_image_and_assistant_only_action(self):
        q,a=training_messages_for_synthetic_contract(
            "حرّك شريط التحكم خطوة واحدة",
            payload({"kind":"key_press","point":[200.5,320.0],"key":"ArrowRight"}),
        )
        self.assertEqual(len(q),1)
        self.assertEqual(q[0]["content"][0],{"type":"image"})
        self.assertIn("حرّك",q[0]["content"][1]["text"])
        self.assertEqual(a[-1]["role"],"assistant")
        self.assertEqual(json.loads(a[-1]["content"][0]["text"])["actions"][0]["key"],"ArrowRight")
        self.assertNotIn("training_supervision",str(q))

    def test_missing_invalid_multimodal_task_rejected(self):
        for task in ("", "x"*501, "\x00", "foo\rbar", None):
            with self.subTest(task=repr(task)),self.assertRaises(ValueError):
                source_task_prompt(task)

    def test_completion_mask_covers_only_assistant_tokens(self):
        labels=assistant_completion_mask([1,2,3],[1,2,3,4,5,6,7,8,9])
        self.assertEqual(labels,[-100,-100,-100,4,5,6,7,8,9])
        self.assertEqual(len(labels),9)

    def test_tokenizer_prefix_drift_and_missing_completion_rejected(self):
        for a,b in (
            ([1,2],[1,3,4,5,6,7,8]),
            ([1,2],[1,2,3,4]),
            ([1],[2,3,4,5,6,7]),
            ([],[1,2,3,4,5]),
            ([True],[True,1,2,3,4,5,6]),
        ):
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):
                assistant_completion_mask(a,b)

    def test_real_original_training_disallowed_even_if_caller_forces_all_flags(self):
        self.assertEqual(len(ADMITTED_TRAIN_MANIFEST_SHAS),0)
        with self.assertRaisesRegex(ValueError,"no independently pinned"):
            verify_real_training_release(release())

    def test_current_research_quarantine_is_never_a_train_release(self):
        for fields in (
            {"official_admitted":False},
            {"split":"DEV"},
            {"complete_old_rgba_proven":False},
            {"complete_source_history_proven":False},
            {"sealed_dev_gold":False},
            {"consumed_ood_reused":True},
            {"manifest_sha256":"g"*64},
            {"unexpected_gold":"leak"},
        ):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                verify_real_training_release(release(**fields))

    def test_synthetic_cpu_contract_never_opens_private_sources_or_model(self):
        result=synthetic_cpu_sft_dry_run()
        self.assertEqual(result["synthetic_action_examples"],3)
        self.assertEqual(result["real_private_images_opened"],0)
        self.assertEqual(result["real_private_gold_opened"],0)
        self.assertEqual(result["real_original_train_cases_admitted"],0)
        self.assertEqual(result["model_calls"],0)
        self.assertEqual(result["gpu_optimizer_steps"],0)
        self.assertFalse(result["old_ood_accessed"])


if __name__=="__main__":
    unittest.main()
