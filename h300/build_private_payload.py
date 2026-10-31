"""H300: exactly one owner-private code-only FREE T4 Kaggle submission.

Never bundle model weights, original source HTML, screenshots, gold,
generated text, private reports, adapters, optimizer state or datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

from h282.build_private_payload import INSTALL_H282

KERNEL_TITLE = "Hala H300 Private Pretrained FieldPort Full12 T4"
KERNEL_ID = "abdulazizshehri/hala-h300-private-pretrained-fieldport-full12-t4"
SOURCE_NAME = "private_pretrained_fieldport_full12.py"
SCRIPT_NAME = "h300_private_pretrained_fieldport_full12.py"
INSTALL = INSTALL_H282.replace(
    "HALA_H282_PRIVATE_FREE_NONCOMMERCIAL",
    "HALA_H300_PRIVATE_NONCOMMERCIAL_FREE_T4",
)
BOOTSTRAP = """import json as _h300_json
import os as _h300_os
import sys as _h300_sys
from pathlib import Path as _h300_Path
try:
"""
BOOTSTRAP_END = """
except Exception as _error:
    if (_h300_sys.platform == "linux"
            and _h300_Path("/kaggle/working").is_dir()
            and _h300_os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            and _h300_os.environ.get(
                "HALA_H300_PRIVATE_NONCOMMERCIAL_FREE_T4") == "1"):
        _file = _h300_Path("/kaggle/working") / (
            "hala-h300-private-fieldport-stage-diagnostic.json")
        if not _file.exists():
            _allowed = {"ValueError","RuntimeError","OSError","ImportError",
                        "ModuleNotFoundError","CalledProcessError",
                        "FileNotFoundError","OutOfMemoryError"}
            _kind = type(_error).__name__
            _file.write_text(_h300_json.dumps({
                "status":"H300_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE",
                "failure_stage":"DEPENDENCIES",
                "exception_class":_kind if _kind in _allowed
                                  else "OTHER_ERROR_CLASS",
                "source_commit":"2babf8c91ffb63dd1123070449d1cbc240ab0859",
                "model_id":"Qwen/Qwen3-VL-2B-Instruct",
                "model_revision":"89644892e4d85e24eaac8bacfd4f463576704203",
                "verified_model_skill_scores":False,
                "verified_optimizer_steps":0,
                "official_train_dev_admitted":False,
                "blind_ood_proven":False,
                "raw_exception_text_model_outputs_media_points_saved":False,
            },sort_keys=True)+"\\n",encoding="utf-8")
    raise RuntimeError("H300 private dependency/bootstrap failed; "
                       "read only allowlisted stage report") from None
"""


def build(source: Path, destination: Path) -> dict:
    if (not isinstance(source, Path) or not source.is_file()
            or source.name != SOURCE_NAME
            or source.parent.name != "h300"
            or not isinstance(destination, Path) or destination.exists()
            or destination.is_relative_to(source.parent)
            or ".git" in destination.parts):
        raise ValueError("H300 fresh code-only private payload path")
    slug = re.sub(r"[^a-z0-9]+","-",KERNEL_TITLE.lower()).strip("-")
    if KERNEL_ID != "abdulazizshehri/" + slug:
        raise ValueError("H300 exact owner private title and slug mismatch")
    body = source.read_text("utf-8")
    future = "from __future__ import annotations\n"
    markers = (
        'SOURCE_SHA = "2babf8c91ffb63dd1123070449d1cbc240ab0859"',
        'SOURCE_BLOB = "14bc9041fabf59d76adb0fc78b6a16d6d53c6e80"',
        'KERNEL_FLAG = "HALA_H300_PRIVATE_NONCOMMERCIAL_FREE_T4"',
        "def clone_original_source(", "def verified_report(",
        "def safe_failure_receipt(", "def write_private_failure(",
        "def run() -> dict:", 'if __name__=="__main__":',
    )
    if body.count(future) != 1 or not all(x in body for x in markers):
        raise ValueError("H300 frozen original trial privacy contract drift")
    assembled = (future + BOOTSTRAP + textwrap.indent(INSTALL,"    ")
                 + BOOTSTRAP_END + body.replace(future,"",1))
    meta = {
        "id":KERNEL_ID,"title":KERNEL_TITLE,
        "code_file":SCRIPT_NAME, "language":"python",
        "kernel_type":"script", "is_private":"true",
        "enable_gpu":"true","enable_tpu":"false",
        "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],"competition_sources":[],
        "kernel_sources":[],"model_sources":[],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination/SCRIPT_NAME).write_text(assembled,encoding="utf-8")
    (destination/"kernel-metadata.json").write_text(
        json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {
        "status":"H300_CODE_ONLY_PRIVATE_NONCOMMERCIAL_FREE_T4",
        "kernel":KERNEL_ID,"python_source_files":1,
        "metadata_files":1,"media_weights_datasets_gold_uploaded":0,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--destination",required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(build(args.source,args.destination),sort_keys=True))
