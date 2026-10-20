"""H45 original event-driven HTML app states, 24 synthetically authored pages.

No production web apps, cloud credentials, external JS, native clicks or real
users. Stable source only; capture is performed in actual headless Edge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h43.factory import author_cases, canonical, digest
from h44.render import page


def dynamic_page(case:dict)->str:
    initial = page(case)
    item=json.dumps(case["target_item"],ensure_ascii=True)
    amount=json.dumps(case["requested_amount"])
    status=("State" if case["language"]=="en" else "الحالة")
    js=f"""
<style>
#h45-status{{position:fixed;bottom:6px;left:16px;right:16px;min-height:30px;
 color:#203a56;background:#e3eefb;border:1px solid #9aafc9;border-radius:7px;
 font:600 13px 'Segoe UI',Tahoma,Arial,sans-serif;padding:5px 10px;direction:ltr}}
</style>
<output id="h45-status" aria-live="polite">{status}: idle</output>
<script>
"use strict";
(() => {{
  const target={item};
  const desired={amount};
  const state={{selected:null,amount:null,confirmed:false,unrelated_changed:false}};
  const status=document.getElementById("h45-status");
  function render() {{
     document.body.dataset.h45Selected=state.selected===null?"null":state.selected;
     document.body.dataset.h45Amount=state.amount===null?"null":String(state.amount);
     document.body.dataset.h45Confirmed=String(state.confirmed);
     document.body.dataset.h45UnrelatedChanged=String(state.unrelated_changed);
     status.textContent="Selected: "+(state.selected===null?"none":state.selected)+
                        " | Quantity: "+(state.amount===null?"none":state.amount)+
                        " | Confirmed: "+(state.confirmed?"yes":"no")+
                        " | Other changed: "+(state.unrelated_changed?"yes":"no");
  }}
  for(const el of document.querySelectorAll("[data-ref]")) {{
    const ref=el.dataset.ref;
    if(el.tagName==="BUTTON") {{
       el.addEventListener("click",() => {{
         if(ref==="row_target") state.selected=target;
         else if(ref==="row_decoy") state.selected=target+"-WRONG";
         else if(ref==="unrelated_button") state.unrelated_changed=true;
         else if(ref==="confirm_button") {{
           if(state.selected===target && state.amount===desired) state.confirmed=true;
         }}
         render();
       }});
    }} else if(el.tagName==="INPUT") {{
       el.addEventListener("input",() => {{
          const raw=el.value;
          const parsed=Number(raw);
          state.amount=raw.length>0 && Number.isInteger(parsed) && parsed>=0 &&
                       parsed<=100 ? parsed : null;
          render();
       }});
    }}
  }}
  render();
}})();
</script>
"""
    if initial.count("</body>")!=1:
        raise ValueError("Original H44 HTML body unexpectedly changed")
    return initial.replace("</body>",js+"</body>")


def plan(root:Path)->dict:
    root=Path(root)
    pages=root/"pages"
    if pages.exists() and any(pages.iterdir()):
        raise FileExistsError("No H45 HTML overwrite")
    pages.mkdir(parents=True,exist_ok=True)
    manifest=author_cases()
    scenes=[]
    for task in manifest["cases"]:
        payload=dynamic_page(task).encode("utf-8")
        filename=task["id"]+".html"
        (pages/filename).write_bytes(payload)
        scenes.append({
            "id":task["id"],"family":task["family"],"split":task["split"],
            "language":task["language"],"goal":task["goal"],
            "expected_trajectory_actions":3,
            "original_html_sha256":hashlib.sha256(payload).hexdigest(),
            "max_actions":task["max_actions"],
        })
    record={"schema":"hala.h45.original-browser-workflows.v1",
            "h43_manifest_sha256":digest(manifest),
            "h44_first_frame_source_only":True,"external_app":False,
            "native_actions":False,"model_inference":False,
            "scenes":scenes}
    path=root/"scenes.json"
    if path.exists():raise FileExistsError("H45 plan exists")
    path.write_bytes(canonical(record))
    return record


if __name__=="__main__":
    print(json.dumps({"scenes":len(plan(Path("h45/assets"))["scenes"])}))
