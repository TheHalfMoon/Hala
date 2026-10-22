"""H62 the collapse probe: it must catch the H60 failure and clear a good model.

The H60 adapter is the known real failure this probe exists for: it produced
well formed JSON and then answered with the same degenerate point every
time. These tests build that failure, build the opposite case, and check
that the probe separates them.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h62.collapse_probe import (
    CONSTANT_COLLAPSE,
    DEGENERATE_DIVERSITY,
    HEALTHY_DIVERSITY,
    INSUFFICIENT_DATA,
    Prediction,
    VISION_IGNORED,
    VISION_INCONCLUSIVE,
    VISION_USED,
    build_report,
    derangement,
    diagnose_collapse,
    load_predictions,
    vision_pathway_probe,
)


def constant_predictions(count=24, point=(320, 180), box=(310, 170, 330, 190)):
    """The H60 shape: well formed output, one degenerate point forever."""
    return [
        Prediction(case_id=f"c{index}", image_id=f"i{index}", instruction="click it",
                   point=point, target_box=box, valid_output=True)
        for index in range(count)
    ]


def grounded_predictions(count=24):
    """A model that reads the image: the answer sits inside the target."""
    predictions = []
    for index in range(count):
        left = 10 + index * 30
        box = (left, 40 + (index % 5) * 20, left + 20, 60 + (index % 5) * 20)
        predictions.append(Prediction(
            case_id=f"c{index}", image_id=f"i{index}", instruction="click it",
            point=(left + 10, 50 + (index % 5) * 20), target_box=box, valid_output=True))
    return predictions


def language_only_predictions(count=24):
    """A model with a spatial prior but no image: a plausible spread that misses."""
    predictions = []
    for index in range(count):
        box = (10 + index * 30, 40 + (index % 5) * 20, 30 + index * 30, 60 + (index % 5) * 20)
        predictions.append(Prediction(
            case_id=f"c{index}", image_id=f"i{index}", instruction="click it",
            point=(320, 180), target_box=box, valid_output=True))
    return predictions


class CollapseDetectionTests(unittest.TestCase):
    def test_the_h60_shape_is_diagnosed_as_a_constant_collapse(self):
        report = diagnose_collapse(constant_predictions())
        self.assertEqual(report["verdict"], CONSTANT_COLLAPSE)
        self.assertEqual(report["distinct_points"], 1)
        self.assertEqual(report["modal_fraction"], 1.0)
        self.assertEqual(report["modal_point"], [320, 180])

    def test_a_grounded_model_is_not_called_a_collapse(self):
        report = diagnose_collapse(grounded_predictions())
        self.assertEqual(report["verdict"], HEALTHY_DIVERSITY)
        self.assertEqual(report["distinct_points"], 24)

    def test_a_model_that_puts_almost_everything_on_one_point_is_degenerate(self):
        predictions = grounded_predictions(20)
        for prediction in predictions[2:]:
            prediction.point = (320, 180)
        report = diagnose_collapse(predictions)
        self.assertEqual(report["verdict"], DEGENERATE_DIVERSITY)
        self.assertGreaterEqual(report["modal_fraction"], 0.9)

    def test_too_few_predictions_are_not_judged(self):
        report = diagnose_collapse(constant_predictions(3))
        self.assertEqual(report["verdict"], INSUFFICIENT_DATA)

    def test_undecodable_answers_are_separated_from_wrong_answers(self):
        predictions = grounded_predictions(12)
        for prediction in predictions[:4]:
            prediction.point = None
            prediction.valid_output = False
        report = diagnose_collapse(predictions)
        self.assertEqual(report["undecodable"], 4)
        self.assertEqual(report["decoded_points"], 8)
        self.assertLess(report["valid_output_rate"], 1.0)

    def test_the_entropy_of_a_constant_distribution_is_zero(self):
        report = diagnose_collapse(constant_predictions())
        self.assertAlmostEqual(report["coordinate_entropy_bits"], 0.0, places=9)

    def test_the_truth_distribution_is_reported_for_comparison(self):
        report = diagnose_collapse(grounded_predictions())
        self.assertEqual(report["truth"]["targets"], 24)
        self.assertGreater(report["truth"]["target_entropy_bits"], 0.0)


class VisionPathwayTests(unittest.TestCase):
    def test_a_grounded_model_is_recognised_as_using_the_image(self):
        report = vision_pathway_probe(grounded_predictions())
        self.assertEqual(report["verdict"], VISION_USED)
        self.assertGreater(report["matching_hits"], report["shuffled_hits"])
        self.assertGreater(report["vision_gain"], 0.0)

    def test_the_h60_shape_is_recognised_as_ignoring_the_image(self):
        # Every prediction is the same point and every target box contains
        # that point, so matching and shuffled score the same.
        predictions = [
            Prediction(case_id=f"c{index}", image_id=f"i{index}", instruction="click it",
                       point=(320, 180), target_box=(310, 170, 330, 190), valid_output=True)
            for index in range(12)
        ]
        report = vision_pathway_probe(predictions)
        self.assertEqual(report["verdict"], VISION_IGNORED)

    def test_a_language_only_model_is_not_credited_with_seeing(self):
        report = vision_pathway_probe(language_only_predictions())
        self.assertEqual(report["verdict"], VISION_INCONCLUSIVE)
        self.assertEqual(report["matching_hits"], 0)
        self.assertEqual(report["shuffled_hits"], 0)

    def test_too_few_predictions_are_inconclusive(self):
        report = vision_pathway_probe(grounded_predictions(2))
        self.assertEqual(report["verdict"], VISION_INCONCLUSIVE)

    def test_the_derangement_moves_every_index_and_is_a_permutation(self):
        order = derangement(12)
        self.assertEqual(sorted(order), list(range(12)))
        self.assertNotEqual(order, list(range(12)))
        for index, mapped in enumerate(order):
            self.assertNotEqual(index, mapped)


class ReportTests(unittest.TestCase):
    def test_the_report_carries_both_answers_and_the_raw_records(self):
        report = build_report(grounded_predictions(), label="h61-adapter")
        self.assertEqual(report["schema"], "hala.h62.collapse-probe.v1")
        self.assertEqual(report["label"], "h61-adapter")
        self.assertEqual(len(report["predictions"]), 24)
        self.assertIn("verdict", report["collapse"])
        self.assertIn("verdict", report["vision_pathway"])

    def test_predictions_round_trip_through_a_jsonl_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [p.to_json() for p in grounded_predictions(6)]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
            loaded = load_predictions(path)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(loaded[0].point, grounded_predictions(6)[0].point)
            self.assertEqual(loaded[0].target_box, grounded_predictions(6)[0].target_box)
            self.assertTrue(loaded[0].valid_output)


if __name__ == "__main__":
    unittest.main()
