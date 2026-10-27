"""H160 owner-private, zero-gold-points second-app exploratory model input.

Freeze H159 16/16 pixel-verified PNGs with preregistered bilingual prompts.
No training, optimizer steps, or H140 blind data. Create-once private pack.
"""
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from h158.original_todomvc_source import tasks_for
from h158.pack_private_ood import SHA_H156, SHA_H157
from h159.stable_source_qa import SCHEMA as H159_SCHEMA

DATASET = "abdulazizshehri/hala-h160-private-stable-secondapp-eval-v1"
SCHEMA = "hala.h160.bilingual-stable-original-todomvc-neural-source.v1"
H159_SHA = "ac3dd44a65349223298b44470fce9dde8de78541f2b5997b003bf72a08ca13c7"
SOURCE_ZIP = "h160-stable-original-todomvc-source.zip"
OLD_ZIP = "h156-frozen-research-lora.zip"
NEW_ZIP = "h157-frozen-research-lora.zip"
CHECKSUMS = {
    OLD_ZIP: "b10e3e32fc4bf d098d551b9b93e14b4bd21d87a9ed5a245975c17d404b47136b".replace(" ", ""),
    NEW_ZIP: "14e3a9a61123a92fb23d53297531831fa19ff7d2435d6704717af7a90c08e808",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def instruction(case):
    language, task = case["language"], case["task"]
    if task == "complete_item":
        target = tasks_for(language, case["index"])[1]
        return (
            f"Click the completion checkbox for '{target}'."
            if language == "en"
            else f"اضغط على مربع إكمال المهمة «{target}»."
        )
    if task == "filter_active":
        return ("Click the Active filter." if language == "en"
                else "اضغط على فلتر المهام النشطة.")
    if task == "clear_completed":
        return ("Click the Clear completed button." if language == "en"
                else "اضغط على زر مسح المهام المكتملة.")
    raise ValueError("H160 unsupported pretrained action")


def prepare(owner):
    owner = Path(owner)
    h159 = owner / "private-h159"
    source = h159 / "source-once-v1"
    frozen = (source / "manifest.json").read_bytes()
    receipt = json.loads((h159 / "H159_SOURCE_REPLAY_QA.json").read_text("utf8"))
    if (sha(frozen) != H159_SHA or
        receipt.get("manifest_sha256") != H159_SHA or
        receipt.get("status") != "H159_16_OF_16_INDEPENDENT_SCREENSHOT_SHA_MATCH"):
        raise ValueError("H160 refuses unverified H159 source")
    source_doc = json.loads(frozen)
    if source_doc.get("schema") != H159_SCHEMA:
        raise ValueError("H160 wrong prior source schema")
    seen = set()
    cases = []
    for c in source_doc["private_cases"]:
        path = source / "images" / c["image"]
        data = path.read_bytes()
        if (sha(data) != c["image_sha256"] or
            c["image_sha256"] in seen or
            c["index"] not in range(301, 317)):
            raise ValueError("H160 original image identity failed")
        seen.add(c["image_sha256"])
        cases.append({
            k: c[k] for k in
            ("task", "language", "index", "image", "image_sha256")
        } | {"instruction": instruction(c)})
    if len(cases) != 16 or len(seen) != 16:
        raise ValueError("H160 16 different untouched source inputs required")
    # Crucially: NO original reference_point is exported.
    packaged = {
        "schema": SCHEMA,
        "upstream": "tastejs/todomvc",
        "upstream_source_sha": source_doc["source_sha"],
        "parent_h159_manifest_sha256": H159_SHA,
        "model_calls": 0, "optimizer_steps": 0,
        "private_cases": cases,
    }

    directory = owner / "private-h160" / "kaggle-eval-v1"
    if directory.exists():
        raise FileExistsError("H160 frozen model eval package is create-once")
    directory.mkdir(parents=True, exist_ok=False)
    with ZipFile(directory / SOURCE_ZIP, mode="x",
                 compression=ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("manifest.json", json.dumps(
            packaged, indent=2, sort_keys=True) + "\n")
        for c in cases:
            z.write(source / "images" / c["image"], "images/" + c["image"])
    with ZipFile(directory / SOURCE_ZIP) as z:
        if (len(z.namelist()) != 17 or z.testzip() is not None or
            any("reference_point" in name for name in z.namelist())):
            raise ValueError("H160 source ZIP incomplete")
    donor = owner / "private-h158" / "kaggle-eval-only-v1"
    for name, expected in CHECKSUMS.items():
        if sha((donor / name).read_bytes()) != expected:
            raise ValueError("H160 prior model ZIP checksum changed")
        shutil.copyfile(donor / name, directory / name)
    metadata = {
        "id": DATASET, "title": DATASET.split("/")[1],
        "licenses": [{"name": "other"}]
    }
    (directory / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", "utf8")
    return {
        "status": "H160_FROZEN_16_PRIVATE_GPU_READY",
        "source_zip_sha256": sha((directory / SOURCE_ZIP).read_bytes()),
        "source_manifest_sha256": sha(json.dumps(
            packaged, indent=2, sort_keys=True).encode() + b"\n"),
        "source_image_count": 16, "model_calls": 0,
        "owner_gold_uploaded": False,
        "h140_blind_calls": 0, "paid_gpu": False,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.owner), indent=2, sort_keys=True))
