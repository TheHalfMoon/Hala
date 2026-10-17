"""Independent CPU-only verification of retrieved H22 Kaggle Saved Version output.

A passing report attests bounded artifact byte integrity and reported identities,
NOT truthful hardware readings, valid science, model quality, release or pricing.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from tools.h22_batch_operator import (FREEZE, SOURCE, MODEL_REV, H19_MANIFEST,
                                      ADAPTER_MANIFEST, MODES, completion_counts)

MAX_ZIP = 16 * 1024**2
MAX_CONTENT = 64 * 1024**2
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_object(raw):
    def no_repeat(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("Duplicate JSON key: " + key)
            data[key] = value
        return data
    def reject_constant(value):
        raise ValueError("Nonfinite JSON literal not permitted: " + value)
    return json.loads(raw, object_pairs_hook=no_repeat, parse_constant=reject_constant)


def saved_file(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, symlinked or oversized retained file: " + path.name)
    return path.read_bytes()


def read_zip(path):
    raw = saved_file(path, MAX_ZIP)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) > 1000 or len(infos) != len({i.filename for i in infos}):
            raise ValueError("Malformed retained evidence inventory")
        if sum(i.file_size for i in infos) > MAX_CONTENT:
            raise ValueError("Unbounded retained evidence uncompressed size")
        for item in infos:
            name = item.filename
            if (not name or name.startswith("/") or chr(92) in name or
                    any(s in ("", ".", "..") for s in name.rstrip("/").split("/")) or
                    item.file_size > 8 * 1024**2 or
                    ((item.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe evidence ZIP member")
        if archive.testzip() is not None:
            raise ValueError("Evidence ZIP CRC failure")
        names = set(archive.namelist())
        minimum = {"profile.log", "profile-status.json", "assigned-hardware.json",
                   "h22_batch_operator.py", "run/result.json"}
        if not minimum.issubset(names):
            raise ValueError("Full measured raw evidence absent")
        contents = {k: archive.read(k) for k in minimum}
    return raw, contents


def verify(results):
    results = Path(results)
    batch = strict_object(saved_file(results / "batch-status.json", 65536))
    assigned_raw = saved_file(results / "assigned-hardware.json", 65536)
    assigned = strict_object(assigned_raw)
    if (batch.get("schema") != "hala.h22.batch_test_triplet.v1" or
            batch.get("status") != "completed_unqualified" or
            batch.get("training_runs") != 0 or batch.get("model_adopted") is not False or
            batch.get("source_commit") != SOURCE or
            batch.get("earlier_int8_outcome") != "unknown" or
            batch.get("earlier_nf4_outcome") != "not_started"):
        raise ValueError("Incorrect complete batch status/previous evidence")
    if (assigned.get("schema") != "hala.h22.batch_assigned_hardware.v1" or
            assigned.get("source_commit") != SOURCE or
            assigned.get("source_freeze_sha256") != FREEZE or
            assigned.get("h19_manifest_sha256") != H19_MANIFEST or
            assigned.get("adapter_manifest_sha256") != ADAPTER_MANIFEST or
            assigned.get("provider") != "kaggle" or
            assigned.get("gpu_name") != "Tesla T4" or
            assigned.get("kernel_run_type") != "Batch" or
            assigned.get("torchao_optional_runtime") != "absent" or
            assigned.get("batch_profile_order") != list(MODES) or
            assigned.get("training_runs") != 0 or assigned.get("no_adoption") is not True):
        raise ValueError("Assigned hardware/identity evidence incomplete")
    gpu = assigned.get("gpu_uuid")
    if not isinstance(gpu, str) or not UUID.fullmatch(gpu):
        raise ValueError("Unrecognized assigned physical GPU UUID")
    if batch.get("assigned_hardware") != assigned:
        raise ValueError("Batch hardware differs from retained assigned hardware")
    profiles = batch.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 3:
        raise ValueError("All three TEST profiles required")
    verified = []
    for index, mode in enumerate(MODES):
        name = f"profile-{index}-{mode}-test"
        row = profiles[index]
        if (type(row) is not dict or row.get("mode") != mode or row.get("split") != "test" or
                row.get("status") != "finished" or row.get("returncode") != 0 or
                row.get("result_status") != "measured" or row.get("gpu_uuid") != gpu):
            raise ValueError("Missing/failed/misattributed TEST mode: " + mode)
        archive_raw, contents = read_zip(results / (name + "-evidence.zip"))
        archive_sha = sha(archive_raw)
        if row.get("archive_sha256") != archive_sha or row.get("archive_bytes") != len(archive_raw):
            raise ValueError("Saved batch evidence ZIP SHA256 mismatch: " + mode)
        if contents["assigned-hardware.json"] != assigned_raw:
            raise ValueError("Per-profile hardware record mismatch")
        expected_operator = Path(__file__).with_name("h22_batch_operator.py").read_bytes()
        if contents["h22_batch_operator.py"] != expected_operator:
            raise ValueError("Saved operator differs from reviewed source")
        status = strict_object(contents["profile-status.json"])
        if status != {k: row[k] for k in status} or status.get("status") != "finished":
            raise ValueError("Per-profile status record mismatch")
        if status.get("mode") != mode or status.get("gpu_uuid") != gpu:
            raise ValueError("Profile metadata mismatch")
        raw_result = contents["run/result.json"]
        if row.get("result_sha256") != sha(raw_result):
            raise ValueError("Raw result SHA256 mismatch")
        result = strict_object(raw_result)
        if (result.get("status") != "measured" or result.get("source_commit") != SOURCE or
                result.get("model_revision") != MODEL_REV or result.get("model_id") != "Qwen/Qwen3-VL-2B-Instruct" or result.get("split") != "test" or
                result.get("gpu_uuid") != gpu or result.get("adapter_loaded") is not True or
                result.get("precision_profile", {}).get("mode") != mode or
                result.get("precision_profile", {}).get("source_freeze_sha256") != FREEZE):
            raise ValueError("Actual model/result source or profile mismatch")
        counts = completion_counts(result)
        if counts != row.get("completed_by_language"):
            raise ValueError("Reported task completion count differs from raw result")
        verified.append({"mode": mode, "archive_sha256": archive_sha,
                         "raw_result_sha256": sha(raw_result),
                         "completed_by_language": counts, "gpu_uuid": gpu})
    return {"schema": "hala.h22.batch_retrieval_audit.v1",
            "saved_version_retrieval_verified": True,
            "all_three_profile_archives_verified": True,
            "assigned_gpu_uuid": gpu, "profiles": verified,
            "source_commit": SOURCE, "training_runs": 0,
            "model_adopted": False, "physical_hardware_truth_independently_verified": False,
            "general_task_quality_qualified": False, "aladdin_qualified": False,
            "scope": "CPU hashes, retained raw result/metadata consistency; not source execution or hardware attestation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved_version_results", help="Downloaded Kaggle Saved Version .../hala-h22-batch/results")
    parser.add_argument("--report", required=True, help="New local JSON report path; must not exist")
    args = parser.parse_args()
    target = Path(args.report)
    if target.exists():
        raise FileExistsError("Never replace an independent audit")
    value = verify(args.saved_version_results)
    target.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
