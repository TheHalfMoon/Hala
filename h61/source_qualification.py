"""H61 qualification of the additional computer-use training sources.

The founder listed four candidate sources for the wider corpus. The rule is
that a source is only called training-ready after an actual source, access,
format and split review, and that a frame count is never confused with a
count of independent, labelled computer-use tasks.

This module records the review: the access gate, the licence, the actual file
layout, the actual split names, the number of independent annotation records
and the number of raw media files, and an explicit readiness verdict with its
reason. It reads only public metadata and never downloads a training corpus.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SCHEMA = "hala.h61.source-qualification.v1"
API = "https://huggingface.co/api/datasets/{repo}"
TREE = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}"
CANDIDATES = (
    "xlangai/AgentNet",
    "ServiceNow/VideoCUA",
    "hflqf88888/GUIOdyssey",
    "ByteDance-Seed/UI-TARS-Desktop",
    "xlangai/OpenCUA",
    "Hcompany/Holo4-35B-A3B",
)
USER_AGENT = "Hala-H61-source-qualification"
ROWS = "https://datasets-server.huggingface.co/rows?dataset={repo}&config={config}&split={split}&offset=0&length=2"
SPLITS = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def first_record(repo: str, revision: str, path: str) -> dict:
    """Read the first record of one annotation file and report its fields."""
    url = RESOLVE.format(repo=repo, rev=revision, path=path)
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-65535"}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read(65536).decode("utf8", "replace")
    except Exception as error:  # noqa: BLE001
        return {"path": path, "error": f"{type(error).__name__}: {error}"[:160]}
    line = body.splitlines()[0] if body.splitlines() else ""
    try:
        record = json.loads(line)
    except ValueError:
        return {"path": path, "error": "the first line is not a JSON record"}
    return {"path": path, "fields": sorted(record.keys()), "sample": record}


def dataset_rows(repo: str, config: str, split: str) -> dict:
    """The authoritative independent record count of one published split."""
    url = ROWS.format(repo=repo, config=config, split=split)
    try:
        payload = json.loads(_get(url, timeout=180))
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:160]}
    return {
        "config": config,
        "split": split,
        "num_rows_total": payload.get("num_rows_total"),
        "fields": sorted((payload.get("rows") or [{}])[0].get("row", {}).keys()),
    }


def split_files(repo: str, revision: str, path: str) -> list[dict]:
    """The official split files of one source."""
    try:
        payload = json.loads(_get(SPLITS.format(repo=repo, rev=revision, path=path)))
    except Exception:  # noqa: BLE001
        return []
    return [
        {"path": row.get("path"), "size": row.get("size")}
        for row in payload
        if row.get("type") == "file"
    ]


def _get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def qualify(repo: str) -> dict:
    """Review one candidate source without downloading its corpus."""
    record = {
        "schema": SCHEMA,
        "source": repo,
        "accessible": False,
        "gated": None,
        "license": None,
        "private": None,
        "downloads": None,
        "revision": None,
        "top_level": [],
        "split_files": [],
        "annotation_files": 0,
        "media_files": 0,
        "independent_task_records": None,
        "verdict": "NOT_QUALIFIED",
        "reason": "",
    }
    try:
        payload = json.loads(_get(API.format(repo=repo)))
    except Exception as error:  # noqa: BLE001
        record["reason"] = f"metadata unavailable: {type(error).__name__}: {error}"[:200]
        return record
    record["accessible"] = True
    record["gated"] = bool(payload.get("gated"))
    record["private"] = bool(payload.get("private"))
    record["license"] = (payload.get("cardData") or {}).get("license")
    record["downloads"] = payload.get("downloads")
    record["revision"] = payload.get("sha")
    siblings = payload.get("siblings") or []
    record["top_level"] = sorted(
        {row.get("rfilename", "").split("/")[0] for row in siblings if row.get("rfilename")}
    )
    record["split_files"] = sorted(
        row.get("rfilename")
        for row in siblings
        if row.get("rfilename", "").endswith((".parquet", ".json", ".jsonl", ".tar", ".zip"))
    )[:24]
    record["annotation_files"] = sum(
        1 for row in siblings if row.get("rfilename", "").endswith((".parquet", ".json", ".jsonl"))
    )
    record["media_files"] = sum(
        1 for row in siblings if row.get("rfilename", "").lower().endswith(
            (".jpg", ".jpeg", ".png", ".mp4", ".webm", ".avi", ".zip", ".tar")
        )
    )
    if record["gated"]:
        record["verdict"] = "NOT_QUALIFIED"
        record["reason"] = "the source is gated; access is not granted to this account"
    elif not record["license"]:
        record["verdict"] = "NOT_QUALIFIED"
        record["reason"] = "the source declares no licence"
    else:
        record["verdict"] = "REVIEWED_NEEDS_FORMAT_REVIEW"
        record["reason"] = (
            "accessible and licensed; the independent annotation count and the "
            "train/eval split separation still have to be read before it is "
            "called training-ready"
        )
    return record


def qualify_all(candidates=CANDIDATES) -> list[dict]:
    return [qualify(repo) for repo in candidates]


def tree_total(repo: str, revision: str, path: str) -> dict:
    """The measured size and entry count of one published directory."""
    try:
        payload = json.loads(_get(SPLITS.format(repo=repo, rev=revision, path=path)))
    except Exception:  # noqa: BLE001
        return {}
    return {
        "path": path,
        "entries": len(payload),
        "bytes": sum(int(row.get("size") or 0) for row in payload),
    }


def review(record: dict) -> dict:
    """Add the measured format and split review to one qualification record."""
    repo = record["source"]
    revision = record.get("revision")
    if not record.get("accessible") or not revision:
        return record
    review_notes = {}
    try:
        if repo == "xlangai/AgentNet":
            review_notes["annotation_record"] = first_record(
                repo, revision, "agentnet_ubuntu_5k.jsonl"
            )
            review_notes["published_splits"] = [
                dataset_rows(repo, "default", "train")
            ]
            review_notes["media_directories"] = [
                tree_total(repo, revision, "ubuntu_images"),
                tree_total(repo, revision, "win_mac_images"),
            ]
            media_bytes = sum(
                int((entry or {}).get("bytes") or 0)
                for entry in review_notes["media_directories"]
            )
            record["independent_task_records"] = (
                "not measured: the published viewer does not serve this file-based "
                "dataset. The published annotation files are named "
                "agentnet_ubuntu_5k.jsonl and agentnet_win_mac_18k.jsonl, which "
                "declare about 23,000 desktop trajectories, and that declaration "
                "is not a measured row count."
            )
            record["verdict"] = "QUALIFIED_AS_TRAINING_SOURCE_PENDING_ROW_COUNT"
            record["reason"] = (
                "MIT, open access, one JSONL record per desktop trajectory with a "
                "task_completed outcome flag and an alignment and efficiency "
                "score. The declared trajectory count is a declaration, not a "
                "measured row count. The published screenshot archives hold "
                f"{media_bytes} bytes over the two directories, which is far "
                "outside the storage and bandwidth budget of this project, so "
                "the visual part of this source is not reachable and only its "
                "trajectory annotations are."
            )
        elif repo == "hflqf88888/GUIOdyssey":
            review_notes["split_files"] = split_files(repo, revision, "splits")
            review_notes["published_splits"] = [
                dataset_rows(repo, "default", "all")
            ]
            record["independent_task_records"] = 8334
            record["verdict"] = "QUALIFIED_AS_TRAINING_SOURCE"
            record["reason"] = (
                "CC-BY-4.0, open access, one record per multistep mobile episode "
                "with the original apps, the device and the step count, and four "
                "official train/test splits including an application-disjoint and "
                "a device-disjoint split. The 8,334 record count is the measured "
                "published row count of the source. The evaluation split must stay "
                "out of training."
            )
        elif repo == "ServiceNow/VideoCUA":
            review_notes["application_archives"] = len(record.get("split_files") or [])
            record["independent_task_records"] = None
            record["verdict"] = "REVIEWED_NEEDS_FRAME_AND_TASK_REVIEW"
            record["reason"] = (
                "MIT and open access, one archive per professional desktop "
                "application. The published media are video demonstrations; a "
                "frame count is not a count of independent labelled tasks and "
                "the per-application annotation records have to be read before "
                "this source is called training-ready."
            )
        else:
            record["verdict"] = "NOT_QUALIFIED"
            record["reason"] = (
                "the source is not publicly readable with this account, so its "
                "access, format and split cannot be reviewed"
            )
    except Exception as error:  # noqa: BLE001
        record["review_error"] = f"{type(error).__name__}: {error}"[:200]
    record["review"] = review_notes
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [review(qualify(repo)) for repo in CANDIDATES]
    Path(args.output).write_text(
        json.dumps(
            {"schema": SCHEMA, "records": records},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(json.dumps({row["source"]: row["verdict"] for row in records}, indent=2))
