"""H305: native browser dispatch accounting, invented counters only.

These are NOT neural model generations, official TRAIN/DEV, or blind OOD.
"""
from __future__ import annotations

import unittest

from h298.preloaded_form_two_stage_policy import (
    _native_dispatch_accounting_valid,
)


class NativeDispatchAccountingTests(unittest.TestCase):
    def outcome(self, turns, dispatched, focus, typed, *, write=False):
        return {
            "native_actions": turns,  # historical policy-turn alias
            "native_dispatched_actions": dispatched,
            "focus_click_dispatched": focus,
            "type_dispatched": typed,
            "source_write_guard_refused": write,
        }

    def test_five_real_source_control_flow_categories(self):
        # A generated abstention or a source-refused type is NOT a write.
        cases = (
            (1, 0, False, False, True, False),  # first abstention
            (1, 1, True, False, False, False),  # wrong first focus
            (2, 1, True, False, True, False),  # second abstention
            (2, 1, True, False, False, True),  # wrong code, no write
            (2, 2, True, True, False, False),  # successful dispatch
        )
        for turns, dispatched, focus, typed, abstained, refused in cases:
            with self.subTest(turns=turns, dispatched=dispatched,
                              abstained=abstained, refused=refused):
                result = self.outcome(turns, dispatched, focus, typed,
                                      write=refused)
                self.assertTrue(_native_dispatch_accounting_valid(
                    result, {"abstained": abstained}))

    def test_forged_dispatch_or_inflated_model_turns_fail_closed(self):
        correct = self.outcome(2, 1, True, False, write=True)
        for key, bad in (
            ("native_dispatched_actions", 2),  # refused write miscounted
            ("native_dispatched_actions", 0),  # click erased
            ("native_dispatched_actions", -1),
            ("native_actions", 1),
            ("native_actions", 3),
            ("native_actions", True),  # bool must not count as int
            ("type_dispatched", True),
            ("focus_click_dispatched", False),
            ("source_write_guard_refused", False),
            ("source_write_guard_refused", 1),
        ):
            with self.subTest(key=key, bad=bad):
                mutated = dict(correct)
                mutated[key] = bad
                self.assertFalse(_native_dispatch_accounting_valid(
                    mutated, {"abstained": False}))

    def test_abstention_cannot_be_counted_as_native_dispatch(self):
        abstain = self.outcome(1, 0, False, False)
        self.assertTrue(_native_dispatch_accounting_valid(
            abstain, {"abstained": True}))
        abstain["native_dispatched_actions"] = 1
        self.assertFalse(_native_dispatch_accounting_valid(
            abstain, {"abstained": True}))
        abstain["native_dispatched_actions"] = 0
        for counters in ({"abstained": 1}, {}, {"abstained": False}):
            self.assertFalse(_native_dispatch_accounting_valid(
                abstain, counters))

    def test_missing_or_nonboolean_evidence_rejected(self):
        good = self.outcome(2, 2, True, True)
        for key in tuple(good):
            broken = dict(good)
            broken.pop(key)
            if key == "source_write_guard_refused":
                # The refusal key is optional for historical source paths.
                self.assertTrue(_native_dispatch_accounting_valid(
                    broken, {"abstained": False}))
            else:
                self.assertFalse(_native_dispatch_accounting_valid(
                    broken, {"abstained": False}))
        for invalid in (None, [], "2", 2):
            self.assertFalse(_native_dispatch_accounting_valid(
                invalid, {"abstained": False}))


if __name__ == "__main__":
    unittest.main()
