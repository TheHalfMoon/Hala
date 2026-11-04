"""Build the standalone H424 Kaggle script from the frozen source H419 helpers."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h419"/"kaggle"/"h419_native_drag_continuation.py"
IMPLEMENTATION=ROOT/"h424"/"kaggle"/"h424_checkpoint_eval_impl.py"
OUTPUT=ROOT/"h424"/"kaggle"/"h424_checkpoint_eval.py"
EXPECTED_SOURCE_SHA="e6db9beb5d8e80eb084e4a078bae7609931be0f7247548bcc87af5ae9e0a0a12"

def build():
    source=SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise ValueError("H419 original frozen image/DEV evaluation source changed")
    src=source.decode("utf8")
    marker="\ndef run_h419():"
    if src.count(marker)!=1:
        raise ValueError("H419 original run function not uniquely found")
    prefix=src.split(marker)[0]
    code=prefix+"\n"+IMPLEMENTATION.read_text("utf8")
    compile(code,str(OUTPUT),"exec")
    if OUTPUT.exists() and OUTPUT.read_text("utf8")!=code:
        raise ValueError("H424 standalone exists but differs; refuse silent overwrite")
    OUTPUT.write_text(code,"utf8")
    print("H424_STANDALONE_SHA256",hashlib.sha256(code.encode()).hexdigest())

if __name__=="__main__":
    build()
