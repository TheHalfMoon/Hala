"""H72 free-T4 supervisor must obey an explicit bounded optimizer budget."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from h65.safe_diagnostic import supervise
from tests.test_h65_safe_diagnostic import FakeChild


class BoundedTrainingTests(unittest.TestCase):
    def test_three_step_smoke_is_forwarded_to_actual_diagnostic(self):
        commands = []
        def runner(command, **kwargs):
            commands.append(command)
            return FakeChild([None, 0])
        with tempfile.TemporaryDirectory() as root:
            report = supervise("hala-h72", "a"*40, Path(root),
                               steps="train,evaluate",
                               stage_sha256="b"*64, stage_source_sha="c"*40, max_steps=3,
                               interval=30, max_seconds=90,
                               runner=runner,
                               snapshot=lambda *args: {"status":"VERIFIED_HOST_STAGING", "step": 1},
                               hardware=lambda _: "T4",
                               sleep=lambda _: None, clock=lambda: 0.0)
        self.assertEqual(report["status"],"COMPLETED_WITH_HOST_EGRESS")
        self.assertEqual(commands[0][commands[0].index("--max-steps")+1],"3")
        self.assertEqual(commands[0][commands[0].index("--stage-sha256")+1],"b"*64)
        self.assertEqual(commands[0][commands[0].index("--stage-source-sha")+1],"c"*40)

    def test_invalid_step_budget_refused_before_any_gpu_action(self):
        with tempfile.TemporaryDirectory() as root:
            child = mock.Mock()
            for step in (0,-1,601,1.5,True):
                with self.subTest(step=step), self.assertRaisesRegex(ValueError,"bounded smoke"):
                    supervise("hala-h72","a"*40,Path(root),max_steps=step,
                              runner=child)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
