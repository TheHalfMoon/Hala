"""H61 pre-flight of the real processor path on the real curriculum rows.

This validates the assumption that the coordinate-focused supervision rests
on: that the completion tokens which carry a digit are exactly the coordinate
tokens of the original action, and that a curriculum row fits inside the
declared token budget. It uses the pinned processor and the pinned tokenizer
and it downloads no model weight.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

from h61.supervision import completion_weights  # noqa: E402
from training.scripts.baseline import MODEL_ID, MODEL_REVISION  # noqa: E402

TOKEN_BUDGET = 4096


def main() -> int:
    from transformers import AutoTokenizer

    rows_path = Path(sys.argv[1])
    root = rows_path.parent
    rows = [json.loads(line) for line in rows_path.read_text("utf8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               trust_remote_code=False)
    checked = 0
    problems = []
    shares: list[list[float]] = []
    for row in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
        completion = row["completion"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(completion_ids)
        if decoded.strip() != completion.strip():
            problems.append({"id": row["id"], "problem": "the completion does not decode back",
                             "decoded": decoded[:120]})
            continue
        weights = completion_weights(completion_ids, tokenizer.decode, 8.0)
        coordinate = [index for index, weight in enumerate(weights) if weight > 1.0]
        if not coordinate:
            problems.append({"id": row["id"], "problem": "no coordinate token was found"})
            continue
        coordinate_text = "".join(
            tokenizer.decode([completion_ids[index]]) for index in coordinate
        )
        digits = [character for character in coordinate_text if character.isdigit()]
        if not digits:
            problems.append({"id": row["id"], "problem": "the coordinate tokens carry no digit"})
            continue
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(completion_ids) > TOKEN_BUDGET:
            problems.append({"id": row["id"], "problem": "the text exceeds the token budget"})
            continue
        checked += 1
        plain_share = len(coordinate) / len(weights)
        weighted_share = (
            sum(weight for weight in weights if weight > 1.0) / sum(weights)
        )
        if checked <= 3:
            print(json.dumps({
                "id": row["id"],
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "coordinate_tokens": len(coordinate),
                "structural_tokens": len(weights) - len(coordinate),
                "coordinate_text": coordinate_text,
                "weights": sorted(set(weights)),
                "coordinate_gradient_share_plain_mean": round(plain_share, 4),
                "coordinate_gradient_share_coordinate_weighted": round(weighted_share, 4),
            }, ensure_ascii=False))
        shares.append([plain_share, weighted_share])
    print(json.dumps({
        "checked": checked,
        "problems": problems[:10],
        "problem_count": len(problems),
        "mean_coordinate_gradient_share_plain_mean": round(
            sum(row[0] for row in shares) / len(shares), 4) if shares else None,
        "mean_coordinate_gradient_share_coordinate_weighted": round(
            sum(row[1] for row in shares) / len(shares), 4) if shares else None,
    }, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
