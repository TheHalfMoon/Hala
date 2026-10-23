"""Remote H72 16-MiB chunk verifier and atomic source ZIP reconstruction."""
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path("/content/hala-h72-parts")
OUT = Path("/content/hala-h67-stage.zip")
SCHEMA = "hala.h72.chunk-transfer.v1"
MAX_CHUNK = 16 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_manifest(root=ROOT):
    file = root / "manifest.json"
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 1048576:
        raise ValueError("missing or oversized source manifest")
    m = json.loads(file.read_text("utf8"))
    if (not isinstance(m, dict) or set(m) != {"schema", "zip_sha256", "git_sha", "zip_bytes", "parts"}
            or m["schema"] != SCHEMA or not re.fullmatch("[0-9a-f]{64}", m["zip_sha256"])
            or not re.fullmatch("[0-9a-f]{40}", m["git_sha"])
            or type(m["zip_bytes"]) is not int or not 0 < m["zip_bytes"] <= 10 * 1024**3
            or not isinstance(m["parts"], list) or not 0 < len(m["parts"]) <= 1000):
        raise ValueError("invalid H72 source manifest")
    offset = 0
    for i, p in enumerate(m["parts"]):
        if (not isinstance(p, dict) or set(p) != {"name", "sha256", "bytes"}
                or p["name"] != f"chunk_{i:05d}.bin"
                or not re.fullmatch("[0-9a-f]{64}", p["sha256"])
                or type(p["bytes"]) is not int or not 0 < p["bytes"] <= MAX_CHUNK
                or (i + 1 < len(m["parts"]) and p["bytes"] != MAX_CHUNK)):
            raise ValueError("untrusted or misordered chunk")
        offset += p["bytes"]
    if offset != m["zip_bytes"]:
        raise ValueError("chunk total differs from original bytes")
    return m


def present(m, root=ROOT):
    accepted = []
    for p in m["parts"]:
        f = root / p["name"]
        if not f.is_symlink() and f.is_file() and f.stat().st_size == p["bytes"] and sha(f) == p["sha256"]:
            accepted.append(p["name"])
    return accepted


def assemble(root=ROOT, out=OUT):
    m = read_manifest(root)
    if out.exists():
        if out.is_symlink() or not out.is_file() or sha(out) != m["zip_sha256"]:
            raise FileExistsError("mismatched remote stage may not be overwritten")
    else:
        if len(present(m, root)) != len(m["parts"]):
            raise ValueError("unverified or missing uploaded chunk")
        if shutil.disk_usage(out.parent).free < m["zip_bytes"] + 64 * 1024**2:
            raise OSError("insufficient remote disk")
        with tempfile.TemporaryDirectory(prefix=".h72-", dir=out.parent) as folder:
            temp = Path(folder) / "stage.zip"
            h = hashlib.sha256()
            amount = 0
            with temp.open("xb") as dst:
                for p in m["parts"]:
                    with (root / p["name"]).open("rb") as src:
                        for part in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(part)
                            h.update(part)
                            amount += len(part)
                dst.flush()
                os.fsync(dst.fileno())
            if amount != m["zip_bytes"] or h.hexdigest() != m["zip_sha256"]:
                raise ValueError("reassembled source ZIP digest mismatch")
            os.replace(temp, out)
    result = {"status": "H72_STAGE_READY", "sha256": m["zip_sha256"],
              "git_sha": m["git_sha"], "bytes": m["zip_bytes"]}
    return result


if __name__ == "__main__":
    action = os.environ.get("H72_ACTION", "")
    ROOT.mkdir(parents=True, exist_ok=True)
    if action == "prepare":
        print("H72_ROOT_PREPARED")
    elif action == "probe":
        print("H72_HAVE_JSON=" + json.dumps(present(read_manifest())))
    elif action == "assemble":
        print("H72_STAGE_READY=" + json.dumps(assemble(), sort_keys=True))
    else:
        raise ValueError("unknown remote transfer action")
