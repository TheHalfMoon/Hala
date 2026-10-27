"""H164 future training admission cannot ingest consumed original-app holdouts."""
import hashlib
import unittest
from unittest.mock import patch

from h164.protocol import (
    MIN_TRAIN,MIN_DEV,MIN_OOD,validate_source_splits
)


def row(i,app,lang="en",action="click"):
    return {"app":app,"task":"do native task",
            "image_sha256":hashlib.sha256(str(i).encode()).hexdigest(),
            "language":lang,"action_type":action}


class H164GeometrySplitGuards(unittest.TestCase):
    def test_final_protocol_minimums(self):
        self.assertEqual((MIN_TRAIN,MIN_DEV,MIN_OOD),(384,96,64))

    @patch("h164.protocol.locked_prior_shas",return_value={"a"*64})
    def test_reject_consumed_model_holdout(self, history):
        with self.assertRaisesRegex(ValueError,"consumed"):
            validate_source_splits(".",[{"app":"new","task":"x",
                "image_sha256":"a"*64,"language":"en",
                "action_type":"click"}],[],[],admission="pilot")

    @patch("h164.protocol.locked_prior_shas",return_value=set())
    def test_reject_duplicate_across_splits(self, history):
        a=row(1,"source")
        with self.assertRaisesRegex(ValueError,"leakage"):
            validate_source_splits(".",[a],[dict(a)],[],admission="pilot")

    @patch("h164.protocol.locked_prior_shas",return_value=set())
    def test_reject_dev_gold_upload(self, history):
        dev=row(2,"test-app")
        dev["owner_gold_click"]={"actions":[{"kind":"click","point":[1,2]}]}
        with self.assertRaisesRegex(ValueError,"gold"):
            validate_source_splits(".",[row(1,"train")],[dev],[],
                                   admission="pilot")

    @patch("h164.protocol.locked_prior_shas",return_value=set())
    def test_reject_ood_app_seen_during_training(self, history):
        with self.assertRaisesRegex(ValueError,"OOD app"):
            validate_source_splits(".",[row(1,"same")],[],
                                   [row(2,"same")],admission="pilot")

    @patch("h164.protocol.locked_prior_shas",return_value=set())
    def test_accept_independent_pilot_without_claiming_trained_model(self,history):
        result=validate_source_splits(
            ".",[row(1,"train","ar","drag")],
            [row(2,"dev")],[row(3,"unseen","ar")],admission="pilot")
        self.assertIn("PILOT",result["status"])
        self.assertEqual(result["optimizer_steps"],0)

    @patch("h164.protocol.locked_prior_shas",return_value=set())
    def test_final_requires_new_apps_and_sufficient_examples(self,history):
        with self.assertRaisesRegex(ValueError,"coverage"):
            validate_source_splits(".",[row(1,"a")],
                                   [row(2,"b")],[row(3,"c")])
