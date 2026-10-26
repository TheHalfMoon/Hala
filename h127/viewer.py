"""Generate an offline-only browser view of the private H127 blind ZIP.

Reviewers need only the blind ZIP, never the separately withheld truth JSON.
No upload, public server, model calls, or external JavaScript.
"""
import hashlib
import json
import re
import zipfile
from pathlib import Path
from h127.review_gate import COUNT, SCHEMA

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hala H127 — Blind Source Review</title>
<style>
body{font-family:system-ui,sans-serif;background:#f4f6f9;color:#17202d;margin:0;padding:20px}
main{max-width:1080px;margin:auto}h1{font-size:1.6rem}p{line-height:1.5}
.panel{background:white;padding:18px;border-radius:10px;margin:14px 0}
img{display:block;max-width:100%;height:auto;border:1px solid #aab4c3}
figure{position:relative;margin:0;width:fit-content;max-width:100%}
.point{position:absolute;width:15px;height:15px;border-radius:50%;border:2px solid white;transform:translate(-50%,-50%);pointer-events:none}
#sMarker{background:#dc2626}#eMarker{background:#2563eb}
button{padding:10px 15px;margin:5px;cursor:pointer}input{padding:9px}
#instruction{font-size:1.2rem;font-weight:600;white-space:pre-wrap}
small{color:#4c596b}#notice{font-weight:600}
</style></head><body><main>
<h1>Hala H127 — Blind GUI Identifiability</h1>
<p>Private original TRAIN screenshots. No reference START/END labels are provided.
Use the screenshot and task instruction only. Click START, then END, in native pixel space.
If unclear, abstain rather than guessing. Do not use AI assistance or look up truth.</p>
<div class="panel">
<label>Reviewer ID (unique, non-sensitive): <input id="reviewer" placeholder="reviewer-001"></label>
<label><input id="attest" type="checkbox"> I completed this independently, without model assistance or ground-truth access.</label>
<p><small>This checkbox is self-attestation, not independent verification of reviewer identity.</small></p>
</div>
<div class="panel"><strong id="counter"></strong>
<p id="instruction"></p><figure id="frame">
<img id="screenshot" alt="Task screenshot"><span class="point" id="sMarker" hidden></span>
<span class="point" id="eMarker" hidden></span></figure>
<p id="notice">Choose START point first.</p>
<button id="clear">Clear points</button>
<button id="located">Save selected points</button>
<button id="ambiguous">Abstain: ambiguous instruction / target</button>
<button id="unreadable">Abstain: unreadable image</button>
</div>
<div class="panel"><button id="previous">Previous</button><button id="next">Next</button>
<button id="download">Download completed blind review JSON</button><p id="feedback"></p></div>
</main><script>
"use strict";
const tasks = __TASKS__;
const packHash = "__PACK_HASH__";
let index = 0;
let start = null;
let end = null;
const answers = Object.create(null);
const $ = (id) => document.getElementById(id);
function marker(id,p) {
  const e=$(id);
  e.hidden = !p;
  if (p) { e.style.left = (p[0] / tasks[index].width * 100)+"%";
           e.style.top = (p[1] / tasks[index].height * 100)+"%"; }
}
function display() {
 const t=tasks[index], prior=answers[t.key];
 $("counter").textContent = "Case "+(index+1)+" / "+tasks.length+
  " — "+Object.keys(answers).length+" saved";
 $("instruction").textContent=t.instruction;
 $("screenshot").src=t.image;
 start=prior && prior.status==="located" ? prior.start : null;
 end=prior && prior.status==="located" ? prior.end : null;
 marker("sMarker",start);marker("eMarker",end);
 $("notice").textContent=prior ? "Saved: "+prior.status+"; new clicks replace coordinates." :
  "Click START, then END, or abstain.";
}
$("screenshot").addEventListener("click",(event)=>{
 const t=tasks[index],rect=event.currentTarget.getBoundingClientRect();
 const x=Math.min(t.width-1,Math.max(0,Math.floor((event.clientX-rect.left)*t.width/rect.width)));
 const y=Math.min(t.height-1,Math.max(0,Math.floor((event.clientY-rect.top)*t.height/rect.height)));
 if(start===null || end!==null){start=[x,y];end=null;} else {end=[x,y];}
 marker("sMarker",start);marker("eMarker",end);
 $("notice").textContent=end===null ? "START set. Choose END." : "START & END chosen. Save.";
});
function save(status) {
 const t=tasks[index];
 if(status==="located" && (!start || !end)) { $("feedback").textContent="Pick both points.";return; }
 answers[t.key]={key:t.key,status:status,start:status==="located"?start:null,
  end:status==="located"?end:null};
 $("feedback").textContent="Saved case "+(index+1)+".";
 if(index<tasks.length-1)index++;
 display();
}
$("located").onclick=()=>save("located");
$("ambiguous").onclick=()=>save("ambiguous");
$("unreadable").onclick=()=>save("unreadable");
$("clear").onclick=()=>{start=null;end=null;marker("sMarker",null);marker("eMarker",null);};
$("previous").onclick=()=>{if(index>0)index--;display();};
$("next").onclick=()=>{if(index<tasks.length-1)index++;display();};
$("download").onclick=()=>{
 const reviewer=$("reviewer").value.trim();
 if(!/^[a-zA-Z0-9_-]{4,80}$/.test(reviewer)) { $("feedback").textContent="Enter a valid reviewer ID.";return; }
 if(!$("attest").checked){$("feedback").textContent="Independent blind attestation required.";return;}
 if(Object.keys(answers).length!==tasks.length){$("feedback").textContent="Finish all 64 cases first.";return;}
 const output={schema:"hala.h127.blind-review.v1",pack_sha256:packHash,
  reviewer_id:reviewer, independent_blind_attestation:true,
  annotations:tasks.map(t=>answers[t.key])};
 const blob=new Blob([JSON.stringify(output,null,2)],{type:"application/json"});
 const url=URL.createObjectURL(blob),link=document.createElement("a");
 link.href=url;link.download="h127-blind-review-"+reviewer+".json";
 link.click();URL.revokeObjectURL(url);
 $("feedback").textContent="Saved review file privately. Do not share with the second reviewer.";
};
display();
</script></body></html>
"""


def build_viewer(pack_path, output_dir):
    pack_path, output_dir = Path(pack_path), Path(output_dir)
    if output_dir.exists():
        raise ValueError("H127 review session output must be new")
    if not pack_path.is_file() or pack_path.stat().st_size > 150_000_000:
        raise ValueError("H127 private blind pack missing or oversized")
    pack_sha = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(pack_path) as archive:
        members = archive.namelist()
        if not members or members.count("tasks.json") != 1 or len(members) != COUNT+1:
            raise ValueError("H127 blind pack must hold 64 images and one task manifest")
        manifest = json.loads(archive.read("tasks.json"))
        if (manifest.get("schema") != SCHEMA or len(manifest.get("tasks",[])) != COUNT or
            manifest.get("label_visibility") != "BLIND"):
            raise ValueError("H127 blind manifest is invalid")
        tasks=manifest["tasks"]
        expected={"tasks.json"}
        keys=set()
        for task in tasks:
            key=task.get("key")
            if (not isinstance(key,str) or not re.fullmatch(r"[a-f0-9]{32}",key) or key in keys or
                not isinstance(task.get("instruction"),str) or
                not isinstance(task.get("width"),int) or
                not isinstance(task.get("height"),int) or
                not 64<=task["width"]<=1024 or not 64<=task["height"]<=640 or
                not re.fullmatch(r"images/"+key+r"\.(png|jpg)",task.get("image",""))):
                raise ValueError("H127 malformed blind task")
            keys.add(key);expected.add(task["image"])
        if set(members)!=expected or len(set(members))!=len(members):
            raise ValueError("H127 ZIP contains unexpected or duplicate members")
        if any(archive.getinfo(name).file_size>5_000_000 for name in expected):
            raise ValueError("H127 reviewer media exceeds private resource bound")
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir/"images").mkdir()
        for task in tasks:
            (output_dir/task["image"]).write_bytes(archive.read(task["image"]))
    safe_json = json.dumps(tasks,ensure_ascii=True).replace("<","\\u003c").replace(">","\\u003e")
    html = HTML.replace("__TASKS__",safe_json).replace("__PACK_HASH__",pack_sha)
    (output_dir/"index.html").write_text(html,encoding="utf8")
    return {"status":"PASS_H127_OFFLINE_BLIND_REVIEWER_VIEW_GENERATED",
            "tasks":len(tasks),"pack_sha256":pack_sha,
            "labels_embedded":False,"human_reviews_completed":False}
