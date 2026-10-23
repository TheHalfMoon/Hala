"""H67 explicit host-side upload to a previously admitted free Colab T4.

Does not create a Colab session, install packages, or run inference. Provider
upload success is only a transport receipt; the remote stage import independently
checks the complete archive SHA and exact source-bound file inventory.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from h67.stage import INDEX, SCHEMA, _check_zip, digest
from h62.run_h61_diagnostic import session_hardware

REMOTE_STAGE = "/content/hala-h67-stage.zip"


def upload(bundle: Path, session: str, sha256: str, pinned_sha: str, *,
           runner=subprocess.run, hardware=session_hardware) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", session):
        raise ValueError("untrusted Colab session name")
    if not re.fullmatch(r"[a-f0-9]{64}", sha256 or ""):
        raise ValueError("explicit expected CPU stage digest required")
    if not re.fullmatch(r"[a-f0-9]{40}", pinned_sha or ""):
        raise ValueError("pinned commit digest required")
    bundle = Path(bundle).resolve()
    if not bundle.is_file() or digest(bundle) != sha256:
        raise ValueError("stage file differs from its reviewed digest")
    import zipfile
    with zipfile.ZipFile(bundle) as z:
        index, _ = _check_zip(z, pinned_sha=pinned_sha)
    if hardware(session) != "T4":
        raise RuntimeError("only an already allocated, operator-approved T4 is supported")
    result = runner(["colab", "upload", "-s", session, str(bundle), REMOTE_STAGE],
                    capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError("stage upload failed; no training should start")
    return {"schema": SCHEMA, "status": "UPLOADED_UNVERIFIED_REMOTE_BYTES",
            "session": session, "remote_path": REMOTE_STAGE,
            "sha256": sha256, "git_sha": pinned_sha,
            "dataset_sha256": index["dataset_sha256"],
            "remote_integrity_verified": False, "training_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--pinned-sha", required=True)
    args = parser.parse_args(argv)
    result = upload(args.bundle, args.session, args.sha256, args.pinned_sha)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
