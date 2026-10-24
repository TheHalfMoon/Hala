"""H99 deterministic diverse-prefix sampler for immutable H71 TRAIN only.

No source bytes or curriculum records are mutated. The sampler is a
training-order experiment, not an official benchmark or a model claim.
Primary balancing is by real (language, family). Secondary cycling covers
drag direction and distance. Repeated screenshots and degenerate target
drags are refused in the selected prefix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re

STRATEGY = "hala.h99.train-only-balanced-family-language-geometry.v1"
SEED = "hala.h99.owner-private-fixed-pilot-20261024"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf8")).hexdigest()


def _point(value):
    if (type(value) is not list or len(value) != 2
            or any(type(x) is not int or x < 0 or x > 32767 for x in value)):
        raise ValueError("H99 needs integer nonnegative original source points")
    return value


def _classify(row):
    if (not isinstance(row, dict) or type(row.get("id")) is not str
            or not row["id"] or row.get("split") != "train"
            or row.get("language") not in ("ar", "en")
            or not isinstance(row.get("family"), str) or not row["family"]
            or type(row.get("image_sha256")) is not str
            or not HEX64.fullmatch(row["image_sha256"])):
        raise ValueError("H99 needs immutable TRAIN-only language/family/media metadata")
    raw = row.get("completion")
    if type(raw) is not str or len(raw) > 2048:
        raise ValueError("H99 requires a short source-bound completion")
    action = json.loads(raw)
    if (not isinstance(action, dict)
            or action.get("type") != "drag"
            or action.get("coordinate_space") != "source_image_pixels"):
        raise ValueError("H99 requires original drag supervision")
    x1, y1 = _point(action.get("start_point"))
    x2, y2 = _point(action.get("end_point"))
    dx, dy = x2 - x1, y2 - y1
    distance = math.hypot(dx, dy)
    if not distance:
        return (row["language"], row["family"]), None
    # Eight angle octants and three displacement scales. Stable across
    # image dimensions; absolute screen quadrants are not assumed here.
    direction = int(((math.atan2(dy, dx) + math.pi / 8) % (2 * math.pi))
                    / (math.pi / 4))
    scale = "short" if distance < 80 else "medium" if distance < 240 else "long"
    return (row["language"], row["family"]), (direction, scale)


@dataclass(frozen=True)
class BalancedPlan:
    indices: tuple[int, ...]
    sha256: str
    summary: dict


def build_balanced_plan(records, count: int, *, seed: str = SEED) -> BalancedPlan:
    if (type(count) is not int or count < 1 or count > 128
            or not isinstance(records, (list, tuple))
            or len(records) < count or len(records) > 100000
            or type(seed) is not str or not seed or len(seed) > 256):
        raise ValueError("H99 bounded immutable source order/step contract")
    groups = defaultdict(lambda: defaultdict(list))
    source_languages, source_families = Counter(), Counter()
    ids = set()
    skipped_degenerate = 0
    for index, record in enumerate(records):
        key, geometry = _classify(record)
        if record["id"] in ids:
            raise ValueError("duplicate H71 train record ID")
        ids.add(record["id"])
        source_languages[key[0]] += 1
        source_families[key[1]] += 1
        if geometry is None:
            skipped_degenerate += 1
            continue
        groups[key][geometry].append((index, record["id"], record["image_sha256"]))
    if not groups or len(groups) > count:
        raise ValueError("H99 study cannot cover all nonempty language/family strata")
    for cells in groups.values():
        for rows in cells.values():
            rows.sort(key=lambda r: (_sha(seed + ":" + r[1]), r[1]))
    group_order = sorted(groups, key=lambda k: _sha(seed + ":group:" + repr(k)))
    cell_order = {
        key: sorted(groups[key], key=lambda g: _sha(seed + ":cell:" + repr((key, g))))
        for key in group_order
    }
    chosen, used_images = [], set()
    used_ids, selections = set(), []
    cell_cursors = {key: 0 for key in group_order}
    selected_languages, selected_families, selected_geometry = Counter(), Counter(), Counter()
    for step in range(count):
        candidate = None
        for outer in range(len(group_order)):
            key = group_order[(step + outer) % len(group_order)]
            cells = cell_order[key]
            for inner in range(len(cells)):
                slot = (cell_cursors[key] + inner) % len(cells)
                bucket = cells[slot]
                pool = groups[key][bucket]
                # Remove already consumed/colliding screenshots to prefer
                # independent visual grounding evidence.
                while pool and (pool[0][1] in used_ids or pool[0][2] in used_images):
                    pool.pop(0)
                if pool:
                    candidate = (key, bucket, slot, pool.pop(0))
                    break
            if candidate:
                break
        if candidate is None:
            raise ValueError("H99 insufficient independent original TRAIN screenshots")
        key, geometry, slot, (index, record_id, image_sha) = candidate
        cell_cursors[key] = (slot + 1) % len(cell_order[key])
        chosen.append(index)
        selections.append(record_id)
        used_images.add(image_sha)
        used_ids.add(record_id)
        selected_languages[key[0]] += 1
        selected_families[key[1]] += 1
        selected_geometry[f"dir{geometry[0]}_{geometry[1]}"] += 1
    if len(chosen) != count or len(used_images) != count:
        raise ValueError("H99 incomplete visual evidence diversification")
    digest = _sha(json.dumps({"schema": STRATEGY, "seed": seed, "ids": selections},
                             ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    summary = {
        "schema": STRATEGY, "requested_steps": count,
        "source_count": len(records),
        "source_languages": dict(sorted(source_languages.items())),
        "source_families": dict(sorted(source_families.items())),
        "selected_languages": dict(sorted(selected_languages.items())),
        "selected_families": dict(sorted(selected_families.items())),
        "selected_geometry": dict(sorted(selected_geometry.items())),
        "source_degenerate_targets_skipped": skipped_degenerate,
        "unique_source_images_selected": len(used_images),
        "unique_train_record_ids_selected": len(used_ids),
        "coverage_fraction": count / len(records),
        "plan_sha256": digest,
        "private_raw_ids_in_summary": False,
    }
    return BalancedPlan(tuple(chosen), digest, summary)
