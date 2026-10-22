"""H62 detect the constant-coordinate collapse and prove the vision is used.

The H60 adapter learned the output contract and then collapsed onto a
constant degenerate point: strict bare JSON validity went 0 to 17 of 48,
geometric grounding stayed at 0 of 24, and real GUI drag task success stayed
at 0 of 48. A collapse that is only discovered by reading a final score is a
collapse that is discovered too late. This module turns it into a cheap,
continuous, machine-checkable signal.

Two independent questions are answered, and they are deliberately separate.

**Is the model collapsing?** A model that always answers the same point has
not learned grounding, whatever its output format looks like. The probe
measures the diversity of the predicted points, the dominance of the modal
point, and the entropy of the predicted coordinate distribution, and it
compares the predicted distribution against the distribution the ground
truth actually has. A model that predicts a constant point while the truth
varies is diagnosed as a collapse even when the task is easy.

**Is the vision pathway actually connected?** A model can produce
well-formed coordinates that are produced by the language prior alone. The
probe measures the gap between matching accuracy, where a prediction is
scored against its own image's target, and shuffled accuracy, where the
same predictions are scored against a different image's target under a
deterministic derangement. A model whose output does not change when the
image changes has a broken visual pathway no matter how good its numbers
look in aggregate.

Everything here is pure, CPU-only and dependency free, so the probe can run
on the raw model outputs of a finished run, on a training sample, or inside
a unit test.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "hala.h62.collapse-probe.v1"

# The verdicts.
CONSTANT_COLLAPSE = "CONSTANT_COLLAPSE"
DEGENERATE_DIVERSITY = "DEGENERATE_DIVERSITY"
HEALTHY_DIVERSITY = "HEALTHY_DIVERSITY"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

VISION_USED = "VISION_USED"
VISION_IGNORED = "VISION_IGNORED"
VISION_INCONCLUSIVE = "VISION_INCONCLUSIVE"

# A prediction closer than this many pixels to the target box centre counts
# as a hit. It is one half of the smallest target box that the corpora use.
HIT_TOLERANCE_PX = 4


@dataclass
class Prediction:
    """One model answer about one image.

    ``point`` is the decoded coordinate in image pixels, or ``None`` when the
    answer could not be decoded at all. A model that answers well formed
    JSON with a wrong constant point and a model that answers nothing are
    different failures, and the probe keeps them apart.
    """

    case_id: str
    image_id: str
    instruction: str
    point: tuple | None = None
    target_box: tuple | None = None
    valid_output: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        """Is the decoded point inside the ground truth box?"""
        if self.point is None or self.target_box is None:
            return False
        left, top, right, bottom = self.target_box
        x, y = self.point
        return left <= x < right and top <= y < bottom

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "image_id": self.image_id,
            "instruction": self.instruction,
            "point": list(self.point) if self.point else None,
            "target_box": list(self.target_box) if self.target_box else None,
            "valid_output": self.valid_output,
            "hit": self.hit,
        }


def _point_key(point) -> tuple:
    return (int(point[0]), int(point[1]))


def coordinate_distribution(predictions: list) -> dict:
    """The shape of the predicted coordinate distribution.

    The report separates three different things that a single accuracy
    number hides: whether the answers are well formed, whether they vary,
    and whether they vary in the right way.
    """
    decoded = [p for p in predictions if p.point is not None]
    total = len(predictions)
    points = [_point_key(p.point) for p in decoded]
    counts: dict = {}
    for point in points:
        counts[point] = counts.get(point, 0) + 1
    distinct = len(counts)
    modal_point = None
    modal_count = 0
    if counts:
        modal_point, modal_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    modal_fraction = (modal_count / len(points)) if points else 0.0
    entropy_bits = 0.0
    if points:
        for count in counts.values():
            probability = count / len(points)
            entropy_bits -= probability * math.log2(probability)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    report = {
        "predictions": total,
        "decoded_points": len(points),
        "undecodable": total - len(points),
        "valid_output_rate": (sum(1 for p in predictions if p.valid_output) / total) if total else 0.0,
        "distinct_points": distinct,
        "distinct_point_rate": (distinct / len(points)) if points else 0.0,
        "modal_point": list(modal_point) if modal_point else None,
        "modal_count": modal_count,
        "modal_fraction": modal_fraction,
        "coordinate_entropy_bits": entropy_bits,
        "x_stddev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "y_stddev": statistics.pstdev(ys) if len(ys) > 1 else 0.0,
        "x_range": (max(xs) - min(xs)) if xs else 0,
        "y_range": (max(ys) - min(ys)) if ys else 0,
    }
    report["mean_point"] = (
        [statistics.fmean(xs), statistics.fmean(ys)] if points else None
    )
    return report


def truth_distribution(predictions: list) -> dict:
    """The shape of the ground truth distribution, for comparison."""
    boxes = [p.target_box for p in predictions if p.target_box is not None]
    centres = []
    for box in boxes:
        left, top, right, bottom = box
        centres.append((int((left + right) // 2), int((top + bottom) // 2)))
    counts: dict = {}
    for centre in centres:
        counts[centre] = counts.get(centre, 0) + 1
    distinct = len(counts)
    entropy_bits = 0.0
    if centres:
        for count in counts.values():
            probability = count / len(centres)
            entropy_bits -= probability * math.log2(probability)
    return {
        "targets": len(boxes),
        "distinct_target_centres": distinct,
        "target_entropy_bits": entropy_bits,
    }


def diagnose_collapse(predictions: list, *, minimum: int = 8) -> dict:
    """Diagnose the constant-coordinate collapse from the raw predictions.

    The verdict is a property of the prediction distribution, not of the
    score. A model can score zero while varying its answers (it is
    grounding the wrong thing) and can score well on a single-target task
    while collapsing (it is not grounding at all). The collapse test asks
    the second question.
    """
    report = coordinate_distribution(predictions)
    truth = truth_distribution(predictions)
    report["truth"] = truth
    decoded = report["decoded_points"]
    if decoded < minimum:
        report["verdict"] = INSUFFICIENT_DATA
        report["reason"] = f"only {decoded} decoded points; {minimum} are needed"
        return report
    if report["distinct_points"] == 1:
        report["verdict"] = CONSTANT_COLLAPSE
        report["reason"] = (
            f"every one of the {decoded} decoded predictions is the same point "
            f"{report['modal_point']}"
        )
        return report
    # The truth varies. A prediction distribution that is dramatically
    # flatter than the truth, or that puts almost all of its mass on one
    # point, is degenerate even though it is not literally constant.
    modal = report["modal_fraction"]
    truth_distinct = truth["distinct_target_centres"]
    if modal >= 0.90 or (truth_distinct >= 3 and report["distinct_points"] <= 2):
        report["verdict"] = DEGENERATE_DIVERSITY
        report["reason"] = (
            f"{modal:.1%} of the decoded predictions share one point while the "
            f"ground truth has {truth_distinct} distinct targets"
        )
        return report
    report["verdict"] = HEALTHY_DIVERSITY
    report["reason"] = (
        f"{report['distinct_points']} distinct predicted points across "
        f"{decoded} decoded predictions, modal share {modal:.1%}"
    )
    return report


def derangement(count: int, offset: int = 1) -> list:
    """A deterministic permutation that moves every index.

    Scoring against a shuffled target answers the question "would this
    answer have been right about a different image?". The permutation is
    deterministic so that two runs of the probe produce the same number.
    """
    if count < 2:
        return list(range(count))
    shift = offset % count
    if shift == 0:
        shift = 1
    return [(index + shift) % count for index in range(count)]


def _hits_against(prediction: Prediction, box) -> bool:
    if prediction.point is None or box is None:
        return False
    left, top, right, bottom = box
    x, y = prediction.point
    return left <= x < right and top <= y < bottom


def vision_pathway_probe(predictions: list, *, minimum: int = 4) -> dict:
    """Test whether the predictions actually depend on the image.

    ``matching`` scores every prediction against its own image's target.
    ``shuffled`` scores the same predictions against a different image's
    target. A model that reads the image has a strictly better matching
    score. A model that ignores the image scores the same on both.
    """
    usable = [p for p in predictions if p.point is not None and p.target_box is not None]
    report = {
        "usable_predictions": len(usable),
        "hit_tolerance_px": HIT_TOLERANCE_PX,
    }
    if len(usable) < minimum:
        report["verdict"] = VISION_INCONCLUSIVE
        report["reason"] = f"only {len(usable)} usable predictions; {minimum} are needed"
        return report
    boxes = [p.target_box for p in usable]
    order = derangement(len(usable))
    matching = sum(1 for p in usable if _hits_against(p, p.target_box))
    shuffled = sum(1 for p, index in zip(usable, order) if _hits_against(p, boxes[index]))
    report["matching_hits"] = matching
    report["shuffled_hits"] = shuffled
    report["matching_accuracy"] = matching / len(usable)
    report["shuffled_accuracy"] = shuffled / len(usable)
    report["vision_gain"] = report["matching_accuracy"] - report["shuffled_accuracy"]
    # How far the answer moves when the image is replaced. A model that
    # ignores the image does not move at all.
    movements = []
    for prediction, index in zip(usable, order):
        if prediction.point is None:
            continue
        other = usable[index]
        if other.target_box is None:
            continue
        left, top, right, bottom = other.target_box
        centre = ((left + right) // 2, (top + bottom) // 2)
        movements.append(math.dist(prediction.point, centre))
    report["mean_shift_px"] = statistics.fmean(movements) if movements else 0.0
    if matching == shuffled and matching > 0:
        report["verdict"] = VISION_IGNORED
        report["reason"] = (
            f"the predictions score the same against their own image "
            f"({matching}) and against a different image ({shuffled})"
        )
    elif matching > shuffled:
        report["verdict"] = VISION_USED
        report["reason"] = (
            f"matching accuracy {matching} exceeds shuffled accuracy {shuffled}; "
            f"the answer depends on the image"
        )
    else:
        report["verdict"] = VISION_INCONCLUSIVE
        report["reason"] = (
            f"matching accuracy {matching} does not exceed shuffled accuracy "
            f"{shuffled}; the probe cannot attribute the answers to the image"
        )
    return report


def build_report(predictions: list, *, label: str = "") -> dict:
    """The complete probe report for one set of raw predictions."""
    return {
        "schema": SCHEMA,
        "label": label,
        "prediction_count": len(predictions),
        "collapse": diagnose_collapse(predictions),
        "vision_pathway": vision_pathway_probe(predictions),
        "predictions": [p.to_json() for p in predictions],
    }


def load_predictions(path: Path) -> list:
    """Read a JSONL file of raw predictions into probe records."""
    records = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        point = raw.get("point")
        box = raw.get("target_box")
        records.append(Prediction(
            case_id=str(raw.get("case_id", "")),
            image_id=str(raw.get("image_id", raw.get("case_id", ""))),
            instruction=str(raw.get("instruction", "")),
            point=(int(point[0]), int(point[1])) if point else None,
            target_box=tuple(int(v) for v in box) if box else None,
            valid_output=bool(raw.get("valid_output", point is not None)),
        ))
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="A JSONL file of raw model predictions")
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = build_report(load_predictions(Path(arguments.predictions)), label=arguments.label)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(text + "\n", encoding="utf8")
    print(text)
