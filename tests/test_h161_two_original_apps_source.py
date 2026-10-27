"""H161 original two-app split, typed native action, and leak guards."""
import inspect
import unittest
from collections import Counter

from h161.original_apps_source import (
    APPS_USED, QUOTAS, all_specs, board_setup, prompt, native_todo, native_board,
    run, prior_shas,
)
from h161.audit_source import audit


class H161SourceDesignTests(unittest.TestCase):
    def test_48_train_24_dev_exact_app_balance(self):
        specs=all_specs()
        self.assertEqual(len(specs),72)
        self.assertEqual(Counter((x["split"],x["app"]) for x in specs),Counter({
            ("TRAIN","todomvc-es6"):24,
            ("TRAIN","react-dnd-kit-board"):24,
            ("DEV","todomvc-es6"):12,
            ("DEV","react-dnd-kit-board"):12,
        }))
        self.assertEqual({s["app"] for s in specs},set(APPS_USED))
        self.assertEqual(len({(x["app"],x["index"]) for x in specs}),72)

    def test_todomvc_all_six_family_language_strata(self):
        specs=all_specs()
        for split,count in (("TRAIN",4),("DEV",2)):
            matched=Counter((x["task"],x["language"]) for x in specs
                            if x["split"]==split and x["app"]=="todomvc-es6")
            self.assertEqual(set(matched.values()),{count})
            self.assertEqual(len(matched),6)

    def test_board_source_labels_and_bilingual_drag(self):
        spec={"index":181023,"language":"ar","task":"drag_todo_progress"}
        label,data=board_setup(spec)
        self.assertIn("181023",label)
        self.assertEqual(data["todo"][0]["id"],181023)
        self.assertIn("اسحب",prompt(spec,label))
        spec["language"]="en"
        self.assertIn("Drag",prompt(spec,label))

    def test_training_never_reads_h160_eval_gold(self):
        source="\n".join(inspect.getsource(x) for x in (
            prior_shas, run, native_todo, native_board, audit
        ))
        self.assertNotIn("H160_32_ORIGINAL_APP_MODEL_NATIVE_TASKS.json",source)
        self.assertNotIn("H140_",source)
        self.assertNotIn("optimizer.step",source)
        self.assertNotIn("model.generate",source)
        self.assertIn("training_supervision",inspect.getsource(run))


if __name__=="__main__":
    unittest.main()
