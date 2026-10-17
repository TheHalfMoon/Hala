"""Generate the exact H22 Kaggle Save & Run All notebook; never submit it.

Outputs a *prospective* batch artifact. Execution requires review and a free GPU.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "tools/h22_batch_operator.py"
DESTINATION = ROOT / "training/platforms/h22-kaggle-batch/h22-kaggle-batch.ipynb"
METADATA = ROOT / "training/platforms/h22-kaggle-batch/kernel-metadata.json"


def cell(kind, value):
    result = {"cell_type": kind, "metadata": {},
              "source": value.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result



FONT_PREFLIGHT = r"""# Restore EXACT qualified font from hash-pinned official Ubuntu package.
import hashlib, io, tarfile, urllib.request
from pathlib import Path

PACKAGE_URL = "https://archive.ubuntu.com/ubuntu/pool/main/f/fonts-dejavu/fonts-dejavu-core_2.37-8_all.deb"
PACKAGE_SHA = "40049660c194f3b8a2541fc7369efebb10e9f94bdac836a2f38fafedd10fa73a"
FONT_SHA = "ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280"
target = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
sha = lambda b: hashlib.sha256(b).hexdigest()
if target.exists():
    assert target.is_file() and not target.is_symlink()
    assert sha(target.read_bytes()) == FONT_SHA, "Existing system font differs from frozen media"
else:
    with urllib.request.urlopen(PACKAGE_URL, timeout=45) as response:
        assert response.geturl() == PACKAGE_URL, "Unexpected archive redirect"
        package = response.read(2 * 1024**2 + 1)
    assert len(package) <= 2 * 1024**2 and sha(package) == PACKAGE_SHA
    assert package.startswith(b"!<arch>\n")
    # Ubuntu's hash-pinned .deb uses data.tar.zst; Python 3.13 tarfile cannot
    # decompress Zstandard. Use the Linux system dpkg-deb to emit an ordinary
    # tar archive; do NOT install the package or execute maintainer scripts.
    import subprocess, tempfile
    found = []
    with tempfile.TemporaryDirectory(prefix="hala-h22-font-") as folder:
        deb = Path(folder) / "font.deb"
        deb.write_bytes(package)
        unpacked = subprocess.run(["dpkg-deb", "--fsys-tarfile", str(deb)],
                                  capture_output=True, check=True, timeout=45).stdout
        assert 0 < len(unpacked) <= 16 * 1024**2
        with tarfile.open(fileobj=io.BytesIO(unpacked), mode="r:") as tar:
            for member in tar.getmembers():
                if member.name.lstrip("./") == "usr/share/fonts/truetype/dejavu/DejaVuSans.ttf":
                    assert member.isfile() and member.size == 759720
                    payload = tar.extractfile(member).read(1024*1024)
                    assert len(payload) == member.size and sha(payload) == FONT_SHA
                    found.append(payload)
    assert len(found) == 1, "Exactly one pinned font asset required"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(found[0])
assert sha(target.read_bytes()) == FONT_SHA, "Qualified font unavailable"
print("HALA_H22_PINNED_FONT_QUALIFIED", FONT_SHA)
"""

def generate():
    raw = OPERATOR.read_bytes()
    source = raw.decode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    marker = "HALA_H22_BATCH_OPERATOR_SHA256"
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}, "accelerator": "GPU",
                     "hala_batch_operator_sha256": sha,
                     "hala_source_commit": "07591f109b6683c9302fcf890259ddf0bde91494",
                     "hala_source_freeze_sha256": "98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b",
                     "hala_adoption_qualified": False},
        "cells": [
            cell("markdown", "# Hala H22 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â prospective TEST batch replication\n\n"
                 "One free Kaggle Save & Run All version. No training, no paid compute, "
                 "no production serving and no change to previous published outcomes. "
                 "The 2026-10-17 earlier FP16 TEST stays independent; prior INT8 is unknown. "
                 "The actual assigned T4 is bound on the batch worker before inference. "
                 "Saved outputs require later retrieval, hashes, independent audits and publication. "
                 "This notebook is NOT release qualification.\n"),
            cell("code", "import os\n"
                 "from pathlib import Path\n"
                 "assert os.name == 'posix' and Path('/kaggle/working').is_dir()\n"
                 "assert os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Batch'\n"
                 "assert not Path('/kaggle/working/hala-h22-batch').exists(), 'Do not overwrite a prior run'\n"
                 "# No model weights downloaded or GPU model allocated in this cell.\n"),
            cell("markdown", "Qualify all pinned packages before model allocation. "
                 "No paid plan, token, service credentials, quotas bypass or auto-upgrade. "
                 "Installed Torch must already match the reviewed GPU environment.\n"),
            cell("code", "%pip uninstall --yes torchao\n"
                 "%pip install --quiet --no-deps transformers==5.16.1 accelerate==1.12.0 "
                 "bitsandbytes==0.50.2 peft==0.20.0 safetensors==0.8.0 Pillow==11.3.0\n"
                 "import importlib.metadata\n"
                  "assert importlib.metadata.version('torch') == '2.11.0+cu128'\n"
                 "try:\n"
                 "    importlib.metadata.version('torchao')\n"
                 "except importlib.metadata.PackageNotFoundError:\n"
                 "    print('HALA_TORCHAO_OPTIONAL_RUNTIME_ABSENT')\n"
                 "else:\n"
                 "    raise RuntimeError('Incompatible optional torchao present after uninstall')\n"),
            cell("code", FONT_PREFLIGHT),
            cell("code", "import hashlib, json, subprocess, sys\n"
                 "from pathlib import Path\n"
                 f"data = {json.dumps(source, ensure_ascii=False)}\n"
                 f"expected = {json.dumps(sha)}\n"
                 "assert hashlib.sha256(data.encode('utf-8')).hexdigest() == expected\n"
                 "path = Path('/kaggle/working/hala-h22-operator.py')\n"
                 "assert not path.exists(), 'Do not overwrite existing operator source'\n"
                 "path.write_bytes(data.encode('utf-8'))\n"
                 "print('OPERATOR_SHA256', expected)\n"
                 "subprocess.run([sys.executable, '-u', str(path)], check=True,\n"
                 "               timeout=3*3700+1800)\n"
                 "print('NO PUBLIC RESULT CLAIM UNTIL VERSION OUTPUT RETRIEVAL AND INDEPENDENT AUDIT')\n")
        ]}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": "abdulazizshehri/hala-h22-test-batch-replication-20261017",
                "title": "Hala H22 TEST batch replication 20261017",
                "code_file": "h22-kaggle-batch.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sha


if __name__ == "__main__":
    print(generate())
