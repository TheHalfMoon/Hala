"""H61 simulation of the remote training preflight on the real curriculum.

This runs the trainer's own media check over EVERY curriculum record and the
trainer's own processor audit over a bounded sample, with the real pinned
processor. It is the closest thing to the remote preflight that can run
without a GPU, and it catches the disagreements between the curriculum and
the trainer before a free session is spent on them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.scripts.train_adapter import check_media, prepare_inputs  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1])
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rows = [json.loads(line) for line in
            (root / "h61_drag_curriculum.jsonl").read_text("utf8").splitlines() if line.strip()]
    print(f"H61_PREFLIGHT records={len(rows)}")
    failures = []
    for index, row in enumerate(rows):
        try:
            check_media(row, root, native_drag_jpeg=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"index": index, "id": row["id"], "error": f"{type(error).__name__}: {error}"[:200]})
            if len(failures) > 5:
                break
    print(f"H61_PREFLIGHT media_checked={len(rows)} failures={len(failures)}")
    for failure in failures[:5]:
        print("  ", json.dumps(failure))
    if failures:
        return 1

    from transformers import AutoProcessor

    from training.scripts.baseline import MODEL_ID, MODEL_REVISION

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                              trust_remote_code=False)
    checked = 0
    for row in rows[:sample]:
        try:
            inputs, labels, weights = prepare_inputs(
                row, processor, root, 4096, native_drag_jpeg=True,
                supervision="coordinate_weighted", coordinate_weight=8.0, return_weights=True,
            )
        except Exception as error:  # noqa: BLE001
            print("  PROCESSOR_FAIL", row["id"], f"{type(error).__name__}: {error}"[:200])
            return 1
        supervised = int((labels != -100).sum())
        heavy = int((weights > 1.0).sum())
        print(f"H61_PREFLIGHT sample id={row['id']} tokens={int(inputs.input_ids.numel())} "
              f"supervised={supervised} coordinate={heavy}")
        checked += 1
    print(f"H61_PREFLIGHT processor_audited={checked} failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
