"""H127 private Kaggle CPU source-bound blind review bundle generator."""
import json
from pathlib import Path

from h61.curriculum import load_export
from h73.kaggle_h73_cpu import ARCHIVE_SHA, H71_CODE_SHA, TRAIN_SHA, locate_attached_train
from h81.staged_resume import admit_stage
from h126.source_audit import propose
from h127.review_gate import EXPECTED_SCHEDULE, make_bundle


def run(scratch, output_dir):
    scratch, output_dir = Path(scratch), Path(output_dir)
    if not str(scratch).startswith("/kaggle/tmp/") or not str(output_dir).startswith("/kaggle/working/"):
        raise ValueError("H127 Kaggle owner-private scratch and outputs only")
    if output_dir.exists():
        raise ValueError("H127 private artifacts must be fresh")
    stage = admit_stage(locate_attached_train(), ARCHIVE_SHA,
                        scratch / "H71", H71_CODE_SHA, rederive=True)
    if stage.get("status") != "RESTORED_TRAIN_STAGE" or stage.get("dataset_sha256") != TRAIN_SHA:
        raise ValueError("H127 original owner-private H71 TRAIN data mismatch")
    root = scratch / "H71/export"
    manifest = json.loads((root / "h61_drag_curriculum_manifest.json").read_text("utf8"))
    rows = load_export(root / "h61_drag_curriculum.jsonl", manifest)
    cohort = propose(rows, root)
    if cohort.summary.get("schedule_sha256") != EXPECTED_SCHEDULE:
        raise ValueError("H127 original TRAIN64 schedule changed since H126 owner CPU audit")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = make_bundle(cohort.cases, root, EXPECTED_SCHEDULE,
                         output_dir / "blind-review-pack.zip",
                         output_dir / "PRIVATE-truth-do-not-share.json")
    result.update(original_H71_dataset_sha256=TRAIN_SHA,
                  source_original_TRAIN_verified=64,
                  previous_2624_source_images_excluded=True,
                  model_skill_qualified=False,
                  no_public_artifacts=True)
    (output_dir / "owner-source-receipt.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", "utf8")
    return result
