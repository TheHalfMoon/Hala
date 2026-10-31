"""H261: four original Git source-family historical collision research.

Only safe aggregate counts leave the owner machine. No original blobs, media,
raw digests, private receipts or sealed scored OOD are published or opened.
This is a denial-only research screen, never dataset admission.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess

UPSTREAM = {
    "h250": "https://github.com/Serkanbyx/simple-to-do-list",
    "h251": "https://github.com/3mmar404/LIfe-Os-Free",
    "h260a": "https://github.com/dacheson/pomodoro-timer",
    "h260b": "https://github.com/dinnocenzo/pomodoro-timer-app",
}
SOURCE = frozenset({".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"})
MEDIA = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico", ".svg",
                   ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".wav"})
TOKEN = re.compile(r"[a-z_][a-z_0-9]*|[0-9]+|[^\s]", re.I)
STATUS = "H261_LOCAL_FOUR_FAMILY_HISTORY_RESEARCH_NOT_ADMITTED"


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H261_DENIED_NO_ADMISSION")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          check=True, timeout=60).stdout


def tree_entries(raw: bytes) -> list[tuple[str, str]]:
    result = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, sep, path = item.partition(b"\t")
        require(bool(sep))
        parts = meta.split()
        require(len(parts) == 3 and parts[1] == b"blob"
                and re.fullmatch(rb"[0-9a-f]{40,64}", parts[2]) is not None)
        ext = Path(path.decode("utf-8", "strict")).suffix.lower()
        if ext in SOURCE | MEDIA:
            result.append((ext, parts[2].decode("ascii")))
    return result


def shingles(data: bytes, width: int = 7) -> frozenset[tuple[str, ...]]:
    require(len(data) <= 200_000)
    words = TOKEN.findall(data.decode("utf-8", "strict").lower())
    if len(words) < 30:
        return frozenset()
    return frozenset(tuple(words[i:i + width]) for i in range(len(words) - width + 1))


def similarity(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def inventory(repo: Path, family: str, pin: str) -> dict:
    require(family in UPSTREAM and repo.is_absolute() and repo.is_dir()
            and not repo.is_symlink() and re.fullmatch(r"[0-9a-f]{40}", pin) is not None)
    require(git(repo, "rev-parse", "HEAD").decode().strip() == pin)
    require(not git(repo, "status", "--porcelain").strip())
    require(git(repo, "remote", "get-url", "origin").decode().strip()
            .removesuffix(".git") == UPSTREAM[family])
    commits = git(repo, "rev-list", "--all").decode().splitlines()
    require(1 <= len(commits) <= 256 and pin in commits)
    sources, media = set(), set()
    for commit in commits:
        for ext, blob in tree_entries(git(repo, "ls-tree", "-rz", commit)):
            (sources if ext in SOURCE else media).add(blob)
    require(len(sources) <= 3000 and len(media) <= 3000)
    shapes = {blob: shingles(git(repo, "cat-file", "blob", blob)) for blob in sources}
    return {"commits": len(commits), "sources": sources, "media": media,
            "shapes": shapes}


def compare(a: dict, b: dict) -> dict:
    best, near = 0.0, 0
    for left, right in itertools.product(a["shapes"].values(), b["shapes"].values()):
        score = similarity(left, right)
        best = max(best, score)
        near += score >= 0.90
    return {"identical_source_blobs": len(a["sources"] & b["sources"]),
            "identical_media_blobs": len(a["media"] & b["media"]),
            "source_token_pairs_ge_90pct": near,
            "max_source_token_jaccard_pct": round(best * 100, 2)}


def audit(repos: dict[str, tuple[Path, str]]) -> dict:
    require(set(repos) == set(UPSTREAM))
    inventories = {n: inventory(repos[n][0], n, repos[n][1]) for n in UPSTREAM}
    pairs = {f"{a}_{b}": compare(inventories[a], inventories[b])
             for a, b in itertools.combinations(UPSTREAM, 2)}
    return {"status": STATUS, "families": 4,
            "reachable_commit_counts": {n: x["commits"] for n, x in inventories.items()},
            "historical_source_blob_counts": {n: len(x["sources"]) for n, x in inventories.items()},
            "historical_media_blob_counts": {n: len(x["media"]) for n, x in inventories.items()},
            "pairwise_local_history": pairs,
            "global_deleted_fork_npm_history_complete": False,
            "transitive_media_rights_proven": False,
            "historical_visual_rgba_complete": False,
            "train_admitted": False, "dev_admitted": False, "ood_admitted": False,
            "model_inference_calls": 0, "optimizer_steps": 0,
            "h173_scored_ood_opened": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in UPSTREAM:
        p.add_argument("--" + name + "-repo", type=Path, required=True)
        p.add_argument("--" + name + "-pin", required=True)
    args = p.parse_args()
    repos = {n: (getattr(args, n + "_repo"), getattr(args, n + "_pin")) for n in UPSTREAM}
    try:
        print(json.dumps(audit(repos), sort_keys=True))
        return 0
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        print('{"status":"H261_DENIED_NO_ADMISSION"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
