"""Build standalone H425-v-H417 384 original H421 counterfactual model evaluation.

Exact H421 SHA-pinned source-only loader/scorer and model generator copied
unchanged. Replaces only H419 disqualified model loader with strict future
H425 actual trained-original parent; no Kaggle GPU is launched by build.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h420/kaggle/h421_native_pair_model_eval.py"
RUN=ROOT/"h425/kaggle/h425_h421_paired_eval_impl.py"
OUT=ROOT/"h425/kaggle/h425_h421_paired_eval.py"
H421_ORIGINAL_SHA="63d4cf8cfd8e324a8f9747312de75eed3c68c2a0357ef338ba06c37601f34838"

def build():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=H421_ORIGINAL_SHA:
        raise ValueError("H425 original 384 counterfactual H421 scorer source SHA drift")
    source=raw.decode("utf8").replace("\r\n","\n")
    a="\ndef h421_trained_h419_adapter():"
    b="\ndef h421_model_case(row):"
    c="\ndef run_h421_eval():"
    if any(source.count(z)!=1 for z in (a,b,c)):
        raise ValueError("H425 frozen original H421 evaluator boundaries changed")
    prefix=source.split(a,1)[0]
    remaining=source.split(b,1)[1].split(c,1)[0]
    script=prefix+b+remaining+"\n"+RUN.read_text("utf8")
    compile(script,str(OUT),"exec")
    if OUT.exists() and OUT.read_text("utf8")!=script:
        raise ValueError("H425 generated real original H421 model evaluation source modified unexpectedly")
    OUT.write_bytes(script.encode("utf8"))
    print("H425_H421_ORIGINAL_MODEL_STANDALONE_SHA",hashlib.sha256(script.encode()).hexdigest())

if __name__=="__main__":
    build()
