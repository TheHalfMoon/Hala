"""H167b independent receipt check, pinned to the isolated PEFT compatibility fix."""
import argparse
import json

from h167.audit_combined import audit as audit_h167
from h167.combined_probe import LANGUAGE_ZIP_SHA

FIXED_SOURCE="ae69beef93521d060a2d8a9f5620785b21f55c1f"

def audit(root):
    result=audit_h167(root,source_commit=FIXED_SOURCE)
    if (result["steps"]!=2 or result["languages"]!=["en","ar"]
        or result["frozen_language_unchanged"] is not True
        or result["new_ood_evaluations"]!=0):
        raise ValueError("H167b frozen language and 2-step scope failed")
    return dict(result,status="H167B_INDEPENDENT_DUAL_ADAPTER_SYNTHETIC_T4_PASS")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    print(json.dumps(audit(parser.parse_args().output),sort_keys=True,indent=2))
