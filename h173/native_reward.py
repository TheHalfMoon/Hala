"""One-shot owner Windows native reward for H173 truly unseen original apps.

FAIL CLOSED: source replay + GPU proposals are frozen before any click.
This script never trains, never opens consumed OOD/DEV media, never passes
reference points to the model, and never re-scores a consumed H173 source.
"""
import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

from h164.native_action_contract import parse_action
from h171.image_integrity import same_exact_png_pixels
from h172.infer_gpu import VISION_WEIGHTS_SHA
from h173.capture_ood import (
    MANIFEST, OWNER, SLIDERS, original_native_scene
)
from h173.infer_gpu import (
    MODEL_PROPOSALS_SCHEMA, OOD_ZIP_SHA
)
from h173.pack_ood import SCHEMA as ZIP_SCHEMA, REPLAY_SCHEMA
from h173.source_preflight import (
    OOD_COUNT, preregister_provisional_cases, verified_original_source
)

REWARD_SCHEMA = "hala.h173.unseen-original-ood-one-shot-native-reward.v1"
LOCK_NAME = "H173_UNSEEN_ORIGINAL_OOD_PERMANENTLY_CONSUMED.lock"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def start_consumption_lock(lock, ood_sha, predictions_sha):
    """Atomically reserve the ONE permitted owner-native OOD scoring attempt.

    Exclusive creation is essential: checking exists() and then write_text()
    is racy when two scoring processes start at the same time. A crash after
    exclusive creation intentionally leaves the benchmark consumed.
    """
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "hala.h173.owner-private-ood-consumption-lock.v1",
        "status": "H173_ONE_SHOT_STARTED_AND_IRREVOCABLY_CONSUMED",
        "source_ood_zip_sha256": ood_sha,
        "gpu_proposals_sha256": predictions_sha,
        "total_browser_rewards_so_far": 0,
    }
    try:
        with lock.open("x", encoding="utf8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError(
            "H173 owner original OOD already started or permanently consumed"
        ) from exc
    return record


def exact_mcnemar(wins, regressions):
    n = wins + regressions
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(
        math.comb(n, k) for k in range(min(wins, regressions) + 1)
    ) / (2**n))


def admit(owner_source, ood_zip, predictions):
    source = Path(owner_source)
    if len(OOD_ZIP_SHA) != 64 or sha(Path(ood_zip).read_bytes()) != OOD_ZIP_SHA:
        raise ValueError("H173 requires pinned newly qualified OOD ZIP")
    source_blob = (source / "manifest.json").read_bytes()
    owner_manifest = json.loads(source_blob)
    receipt_raw = (source / "owner-source-replay-receipt.json").read_bytes()
    replay = json.loads(receipt_raw)
    gold = json.loads(
        (source / "owner-reference-private.json").read_text("utf8")
    )
    with zipfile.ZipFile(ood_zip) as archive:
        packed = json.loads(archive.read("manifest.json"))
        expected_members = {"manifest.json"} | {
            "images/" + str(c["index"]) + ".png"
            for c in preregister_provisional_cases()
        }
        names = archive.namelist()
        if len(names) != len(expected_members) or set(names) != expected_members:
            raise ValueError("H173 OOD ZIP contains duplicate files, non-source or owner gold")
        # Never trust that a ZIP's manifest SHA describes its actual pixels.
        # Independently bind all 32 GPU input PNG bytes to owner original source.
        packaged_sha = {
            name: sha(archive.read(name))
            for name in names if name != "manifest.json"
        }
    if (
        owner_manifest.get("schema") != MANIFEST
        or owner_manifest.get("count") != OOD_COUNT
        or owner_manifest.get("split") != "OOD"
        or owner_manifest.get("owner_gold_uploaded") is not False
        or owner_manifest.get("no_old_train_dev_ood_images") is not True
        or owner_manifest.get("no_model_calls_before_frozen_capture") is not True
        or replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != "H173_PRIVATE_NEW_OOD_32_EXACT_PIXEL_REPLAY_PASS"
        or replay.get("cases") != OOD_COUNT
        or replay.get("independent_rgba_pixel_replays") != OOD_COUNT
        or replay.get("source_manifest_sha256") != sha(source_blob)
        or replay.get("model_calls") != 0
        or replay.get("optimizer_steps") != 0
        or packed.get("schema") != ZIP_SCHEMA
        or packed.get("count") != OOD_COUNT
        or packed.get("split") != "OOD"
        or packed.get("owner_gold_uploaded") is not False
        or packed.get("owner_source_manifest_sha256") != sha(source_blob)
        or packed.get("independent_rgba_replay_receipt_sha256")
        != sha(receipt_raw)
        or gold.get("schema") != OWNER
    ):
        raise ValueError("H173 paired OOD owner-only source/replay proof invalid")
    plan = preregister_provisional_cases()
    scenes = owner_manifest.get("private_cases")
    refs = gold.get("owner_reference_actions")
    packed_scenes = packed.get("private_cases")
    if not all(
        isinstance(arr, list) and len(arr) == OOD_COUNT
        for arr in (scenes, refs, packed_scenes)
    ):
        raise ValueError("H173 32 source/gold/packed scenes required")
    by_index = {}
    for spec, row, pack, ref in zip(plan, scenes, packed_scenes, refs):
        ix = spec["index"]
        if (
            any(row.get(k) != v for k, v in spec.items() if k != "split")
            or row.get("split") != "OOD"
            or row.get("original_reference_success") is not True
            or pack.get("index") != ix
            or pack.get("image") != row.get("image")
            or pack.get("image_sha256") != row.get("image_sha256")
            or packaged_sha.get(row["image"]) != row.get("image_sha256")
            or pack.get("instruction") != row.get("instruction")
            or ref.get("index") != ix
            or ref.get("kind") != "click"
            or ref.get("original_native_reference_success") is not True
            or len(row.get("image_sha256", "")) != 64
            or sha((source / row["image"]).read_bytes()) != row["image_sha256"]
        ):
            raise ValueError("H173 original scene/gold/source identity mismatch")
        point = ref.get("point")
        state = ref.get("prestate")
        if (
            type(point) is not list or len(point) != 2
            or any(type(v) is not int or not 0 <= v < bound
                   for v, bound in zip(point, (960, 720)))
            or type(state) is not dict
        ):
            raise ValueError("H173 missing separate real owner click evidence")
        if row["app"] == "tingle":
            if state != {"kind": "modal_visible", "value": True}:
                raise ValueError("H173 native modal reference prestate invalid")
        else:
            if (
                state.get("kind") != "carousel_index"
                or type(state.get("value")) is not int
                or type(state.get("reference_after_index")) is not int
                or state["value"] == state["reference_after_index"]
                or state.get("direction") != (
                    "prev" if row["variant"] % 2 == 0 else "next"
                )
                or state.get("name") != SLIDERS[row["variant"] // 2]
            ):
                raise ValueError(
                    "H173 expected exact original carousel direction receipt"
                )
        by_index[ix] = (row, ref)
    report = json.loads(Path(predictions).read_text("utf8"))
    if (
        report.get("schema") != MODEL_PROPOSALS_SCHEMA
        or report.get("status")
        != "H173_PRIVATE_64_REAL_GPU_UNSEEN_OOD_PROPOSALS_COMPLETE"
        or report.get("source_ood_zip_sha256") != OOD_ZIP_SHA
        or report.get("trained_vision_sha256") != VISION_WEIGHTS_SHA
        or report.get("real_gpu_calls") != OOD_COUNT * 2
        or report.get("paired_ood_cases") != OOD_COUNT
        or report.get("optimizer_steps") != 0
        or report.get("owner_gold_on_gpu") is not False
        or report.get("previous_consumed_dev_ood_used") is not False
        or report.get("paid_gpu") is not False
        or report.get("weights_published") is not False
    ):
        raise ValueError("H173 actual frozen GPU paired OOD evidence invalid")
    proposals = report.get("predictions")
    if not isinstance(proposals, list) or len(proposals) != OOD_COUNT * 2:
        raise ValueError("H173 exactly 64 model responses required")
    by_pair = {}
    for item in proposals:
        key = (item.get("case_index"), item.get("role"))
        row = by_index.get(key[0], (None,))[0]
        if (
            row is None
            or key[1] not in ("before", "after")
            or key in by_pair
            or item.get("image_sha256") != row["image_sha256"]
            or item.get("app") != row["app"]
            or item.get("language") != row["language"]
            or item.get("action_type") != row["action_type"]
            or item.get("parsed") != parse_action(item.get("raw"))
        ):
            raise ValueError("H173 duplicate, mismatched, or forged model output")
        by_pair[key] = item
    if set(by_pair) != {
        (c["index"], role)
        for c in plan
        for role in ("before", "after")
    }:
        raise ValueError("H173 paired model prediction coverage incomplete")
    return plan, by_index, by_pair


def run(owner, upstream, source, ood_zip, predictions, output):
    from playwright.sync_api import sync_playwright
    from h130.browser_env import _chromium_binary
    from h138.probe import serve

    owner, output = Path(owner), Path(output)
    lock = owner / "private-h173" / LOCK_NAME
    if lock.exists() or output.exists():
        raise ValueError("H173 original OOD already attempted or consumed")
    plan, rows, proposals = admit(source, ood_zip, predictions)
    # The native scorer must not accept arbitrary prediction JSON.
    # Independently verify the pinned source commit, FREE T4 job receipt,
    # frozen adapters, complete proposals and their exact immutable hashes
    # before writing the irreversible owner OOD consumption lock.
    from h173.audit_gpu import audit as independently_audit_gpu
    real_gpu = independently_audit_gpu(Path(predictions).parent.parent)
    if (
        real_gpu.get("status")
        != "H173_INDEPENDENT_64_PRIVATE_FREE_T4_UNSEEN_OOD_PROPOSALS_VERIFIED"
        or real_gpu.get("model_calls") != OOD_COUNT * 2
        or real_gpu.get("native_original_browser_reward_measured") is not False
    ):
        raise ValueError("H173 genuine independent FREE T4 proposals not qualified")
    verified_original_source(upstream)
    # Atomic exclusive creation BEFORE the first browser/model action.
    # Interrupted or concurrent attempts must never re-use the benchmark.
    start_consumption_lock(
        lock, OOD_ZIP_SHA, sha(Path(predictions).read_bytes())
    )
    outcomes, counts = [], Counter()
    with serve(Path(upstream)) as url:
        with sync_playwright() as playwright:
            opts = {"headless": True}
            exe = _chromium_binary()
            if exe:
                opts["executable_path"] = exe
            browser = playwright.chromium.launch(**opts)
            try:
                for spec in plan:
                    row, gold = rows[spec["index"]]
                    frozen = (Path(source) / row["image"]).read_bytes()
                    expected_after = gold["prestate"].get(
                        "reference_after_index"
                    )
                    for role in ("before", "after"):
                        item = proposals[(spec["index"], role)]
                        rendered, task, _, before, result = original_native_scene(
                            browser, url, spec,
                            execute_reference=False,
                            model_action=item["parsed"],
                            frozen_image=frozen,
                            expected_after_index=expected_after,
                        )
                        if (
                            not same_exact_png_pixels(frozen, rendered)
                            or task != row["instruction"]
                            or before["kind"] != gold["prestate"]["kind"]
                            or before["value"] != gold["prestate"]["value"]
                            or (
                                spec["app"] == "tiny-slider"
                                and (
                                    before.get("direction")
                                    != gold["prestate"]["direction"]
                                    or before.get("name")
                                    != gold["prestate"]["name"]
                                )
                            )
                        ):
                            raise ValueError(
                                "H173 native screenshot or original prestate drift"
                            )
                        outcome = {
                            "index": spec["index"],
                            "role": role,
                            "app": spec["app"],
                            "language": spec["language"],
                            "image_sha256": row["image_sha256"],
                            "success": result["success"],
                            "valid_click": result["valid_click"],
                        }
                        outcomes.append(outcome)
                        counts[role + "/calls"] += 1
                        counts[role + "/success"] += int(result["success"])
                        counts[role + "/" + spec["app"] + "/success"] += int(
                            result["success"]
                        )
                        counts[role + "/" + spec["language"] + "/success"] += int(
                            result["success"]
                        )
                        print(
                            "H173_ONE_SHOT_REAL_UNSEEN_OOD",
                            len(outcomes), OOD_COUNT * 2, role,
                            result["success"], flush=True
                        )
            finally:
                browser.close()
    paired = {
        (item["index"], item["role"]): item["success"]
        for item in outcomes
    }
    wins = sum(
        paired[(c["index"], "after")] and not paired[(c["index"], "before")]
        for c in plan
    )
    regressions = sum(
        paired[(c["index"], "before")] and not paired[(c["index"], "after")]
        for c in plan
    )
    if (
        len(outcomes) != 64 or counts["before/calls"] != 32
        or counts["after/calls"] != 32
    ):
        raise ValueError("H173 one-shot original browser actions incomplete")
    report = {
        "schema": REWARD_SCHEMA,
        "status": "H173_ONE_SHOT_GENUINELY_UNSEEN_ORIGINAL_OOD_CONSUMED",
        "cases": 32,
        "real_original_native_browser_rewards": 64,
        "source_ood_zip_sha256": OOD_ZIP_SHA,
        "gpu_proposals_sha256": sha(Path(predictions).read_bytes()),
        "trained_vision_sha256": VISION_WEIGHTS_SHA,
        "before_success": counts["before/success"],
        "after_success": counts["after/success"],
        "paired_wins": wins,
        "paired_regressions": regressions,
        "two_sided_exact_mcnemar_p": exact_mcnemar(wins, regressions),
        "counts": dict(counts),
        "private_outcomes": outcomes,
        "real_native_state_not_gold_coordinate_proximity": True,
        "owner_gold_on_gpu": False,
        "old_consumed_eval_used": False,
        "optimizer_steps": 0,
        "paid_gpu": False,
        "flagship_35b_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    lock.write_text(json.dumps({
        "schema": "hala.h173.owner-private-ood-consumption-lock.v1",
        "status": "H173_ONE_SHOT_FINAL_PERMANENTLY_CONSUMED",
        "source_ood_zip_sha256": OOD_ZIP_SHA,
        "gpu_proposals_sha256": report["gpu_proposals_sha256"],
        "total_browser_rewards_so_far": 64,
        "private_reward_sha256": sha(output.read_bytes()),
    }, sort_keys=True) + "\n", encoding="utf8")
    return {key: val for key, val in report.items() if key != "private_outcomes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("owner", "upstream", "source", "ood-zip",
                "predictions", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.owner, args.upstream, args.source, args.ood_zip,
            args.predictions, args.output),
        indent=2, sort_keys=True
    ))
