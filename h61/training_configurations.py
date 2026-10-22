"""H61 the staged training configurations, with the reason for each number.

The first run is a diagnostic of the hypothesis: does the coordinate-focused
supervision move the grounding at all? It therefore keeps the learning rate,
the rank and the coordinate weight of the H60 run and changes only the
supervision, the frame, the contract and the Arabic family, so that a
measured change is attributable.

The staged configuration is the same recipe at a larger scale, with the
processor audit bounded because a full audit of a sixty four thousand row
curriculum would spend hours before the first optimizer step.
"""
import json
from pathlib import Path

SCHEMA = "hala.h61.training-configurations.v1"
REPO = Path(__file__).resolve().parents[1]
FIRST_RUN = "experiments/runtime/h61-colab-train-config.json"
STAGED_RUN = "experiments/runtime/h61-stage-c-train-config.json"


def describe() -> dict:
    first = json.loads((REPO / FIRST_RUN).read_text("utf8"))
    staged = json.loads((REPO / STAGED_RUN).read_text("utf8"))
    return {
        "schema": SCHEMA,
        "first_run": {
            "config": FIRST_RUN,
            "role": "the diagnostic of the supervision hypothesis",
            "changed_against_h60": [
                "the coordinate tokens carry a higher loss weight",
                "the training screenshots are in the evaluation frame",
                "one shared prompt contract for training and evaluation",
                "a bilingual family is mixed into the curriculum",
            ],
            "unchanged_against_h60": [
                "the learning rate",
                "the LoRA rank",
                "the precision",
                "the model and its pinned revision",
            ],
            "samples_seen": first["max_steps"] * first["gradient_accumulation"],
            "values": first,
        },
        "staged_run": {
            "config": STAGED_RUN,
            "role": "the same recipe at the Stage C scale",
            "samples_seen": staged["max_steps"] * staged["gradient_accumulation"],
            "audit_note": (
                "the processor audit is bounded because a full audit of a sixty four "
                "thousand row curriculum would cost about two hours before the first "
                "optimizer step, and the media digest and the format of every record "
                "are still checked"
            ),
            "values": staged,
        },
    }


if __name__ == "__main__":
    print(json.dumps(describe(), ensure_ascii=False, sort_keys=True, indent=2))
