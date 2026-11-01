"""H306: invented adversarial native transition evidence, not model skill."""
import unittest

from h298.preloaded_form_two_stage_policy import (
    _native_dispatch_accounting_valid,
)


def record(turns, dispatches, focus, typed, *, abstained=False,
           focus_refused=False, write_refused=False):
    return ({
        "native_actions": turns,
        "native_dispatched_actions": dispatches,
        "focus_click_dispatched": focus,
        "type_dispatched": typed,
        "source_focus_guard_refused": focus_refused,
        "source_write_guard_refused": write_refused,
    }, {"abstained": abstained})


class ClosedNativeTransitions(unittest.TestCase):
    def test_five_realizable_control_paths(self):
        paths = (
            record(1, 0, False, False, abstained=True),
            record(1, 1, True, False, focus_refused=True),
            record(2, 1, True, False, abstained=True),
            record(2, 1, True, False, write_refused=True),
            record(2, 2, True, True),
        )
        for outcome, counters in paths:
            with self.subTest(outcome=outcome):
                self.assertTrue(
                    _native_dispatch_accounting_valid(outcome, counters))

    def test_impossible_type_without_focus_and_double_refusals(self):
        # All these tuples passed the old H305 arithmetic-only checker.
        forged = (
            record(1, 1, False, True),
            record(2, 1, False, True, abstained=True),
            record(2, 1, True, False, write_refused=True,
                   focus_refused=True),
            record(1, 0, False, False, abstained=True,
                   focus_refused=True),
            record(2, 0, False, False, abstained=True,
                   write_refused=True),
        )
        for outcome, counters in forged:
            with self.subTest(outcome=outcome):
                self.assertFalse(
                    _native_dispatch_accounting_valid(outcome, counters))

    def test_malformed_refusal_flags_are_not_booleans(self):
        good, counters = record(2, 1, True, False, write_refused=True)
        for key in ("source_focus_guard_refused",
                    "source_write_guard_refused"):
            for bad in (1, None, "false"):
                with self.subTest(key=key, bad=bad):
                    modified = dict(good)
                    modified[key] = bad
                    self.assertFalse(
                        _native_dispatch_accounting_valid(modified, counters))

    def test_second_stage_requires_first_real_click(self):
        for outcome, counters in (
            record(2, 1, False, True, write_refused=True),
            record(2, 1, False, True, abstained=True),
            record(2, 1, True, False, abstained=True,
                   focus_refused=True),
        ):
            self.assertFalse(
                _native_dispatch_accounting_valid(outcome, counters))


if __name__ == "__main__":
    unittest.main()
