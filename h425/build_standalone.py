"""Generate the standalone H425 genuine-image LoRA run from frozen H422 helpers.

No H419 final adapter is used: H419 9452-step source research gate FAILED.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h422"/"kaggle"/"h422_two_prompt_native_lora.py"
EXPERIMENT=ROOT/"h425"/"kaggle"/"h425_h417_replay_paired_impl.py"
OUTPUT=ROOT/"h425"/"kaggle"/"h425_h417_replay_paired.py"
PINNED_SOURCE_SHA="6ea306fe3f4cb6ad6044956aecd816e212d253901ebcbc7c1800378db59321de"

def build():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINNED_SOURCE_SHA:
        raise ValueError("H425 frozen H422 original H60+H417 source helpers SHA drift")
    source=raw.decode("utf8").replace("\r\n","\n")
    marker="\ndef run_h422():"
    if source.count(marker)!=1:
        raise ValueError("H422 original source evaluation prefix ambiguous")
    code=source.split(marker)[0]+EXPERIMENT.read_text("utf8")
    compile(code,str(OUTPUT),"exec")
    if OUTPUT.exists() and OUTPUT.read_text("utf8")!=code:
        raise ValueError("H425 generated standalone diverged; delete before explicit regeneration")
    OUTPUT.write_bytes(code.encode("utf8"))
    print("H425_STANDALONE_SHA",hashlib.sha256(code.encode()).hexdigest())

if __name__=="__main__":
    build()
