"""H57: bounded authentic DragOn TRAIN image+task pairs across three NEW domains.

Read exact 206 byte ranges of pinned original H Company TARs, independently
join each JPEG+task by source UUID and original image_id, validate geometry,
preserve ordered=true/false and subtype, and store source bytes PRIVATE only.
Never use public eval or benchmark test, never train or invoke a model.
"""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.request
from PIL import Image

REV="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
SOURCE="Hcompany/DragOn"
WINDOW=64*1024*1024
DOMAINS={
 "sheet":{"total":14258661376,"task_start":13803454464},
 "slide_resize":{"total":9915637248,"task_start":9329180672},
 "text_highlight":{"total":29874024960,"task_start":29211230208},
}
UUID=re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
SCHEMA="hala.h57.hcompany-original-train-multidomain.v1"
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode("utf8")

def bounded_range(domain,part):
    if domain not in DOMAINS or part not in ("images","tasks"):
        raise ValueError("Only frozen official original TRAIN domains are allowed")
    total=DOMAINS[domain]["total"]
    start=0 if part=="images" else DOMAINS[domain]["task_start"]
    if start%512 or start+WINDOW>=total:raise ValueError("Unqualified TAR range")
    url=f"https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/train/{domain}.tar?download=true"
    request=urllib.request.Request(url,headers={
      "Range":f"bytes={start}-{start+WINDOW-1}",
      "User-Agent":"Hala-H57-bounded-official-train-only-2026-10-21"})
    with urllib.request.urlopen(request,timeout=140) as response:
        expected=f"bytes {start}-{start+WINDOW-1}/{total}"
        if (response.status!=206 or response.headers.get("Content-Range")!=expected or
            response.headers.get("Content-Length")!=str(WINDOW)):
            raise ValueError("Source provider did not honor exact immutable TRAIN Range")
        data=response.read(WINDOW+1)
    if len(data)!=WINDOW:raise ValueError("Unbounded or truncated upstream source")
    return data,dict(source=SOURCE,revision=REV,train_tar=f"train/{domain}.tar",
            domain=domain,part=part,start=start,length=WINDOW,
            content_range=expected,sha256=sha(data),full_source_tar_verified=False)

def tar_members(data,start,domain,part):
    if domain not in DOMAINS or part not in ("images","tasks") or len(data)!=WINDOW:
        raise ValueError("TAR parsed only from actual frozen-size source byte range")
    ext=".jpg" if part=="images" else ".json"
    prefix=f"{domain}/{part}/"
    out={}
    for offset in range(0,len(data)-512,512):
        b=data[offset:offset+512]
        try:
            if b[257:262]!=b"ustar":continue
            if sum(b[:148])+256+sum(b[156:])!=int(b[148:156].strip(b" \0") or b"0",8):
                continue
            fn=b[:100].split(b"\0")[0].decode("ascii")
            size=int(b[124:136].strip(b" \0") or b"0",8)
            if (not fn.startswith(prefix) or not fn.endswith(ext) or
                "/._" in fn or "/PaxHeader/" in fn or b[156:157] not in (b"0",b"\0") or
                size<=0 or size>3_000_000 or offset+512+size>len(data)):
                continue
            key=fn[len(prefix):-len(ext)]
            if not UUID.fullmatch(key):continue
            if key in out:raise ValueError("TAR has duplicate real member UUID")
            blob=data[offset+512:offset+512+size]
            out[key]=dict(name=fn,payload=blob,sha256=sha(blob),
                          offset=start+offset+512,size=size)
        except (UnicodeError,ValueError) as exc:
            if "duplicate real member" in str(exc):raise
            continue
    return out

def verify_pair(domain,key,image,original_task_json):
    if domain not in DOMAINS or not UUID.fullmatch(key):
        raise ValueError("Invalid original TRAIN domain/UUID")
    if not (image[:3]==b"\xff\xd8\xff" and image[-2:]==b"\xff\xd9"):
        raise ValueError("Incomplete original JPEG")
    with Image.open(BytesIO(image)) as pic:
        if pic.format!="JPEG":raise ValueError("Not original JPEG")
        width,height=pic.size
        pic.verify()
    if not (300<=width<=5000 and 300<=height<=5000 and width*height<=12_000_000):
        raise ValueError("Image dimensions exceed original-source budget")
    source=json.loads(original_task_json.decode("utf8"))
    if not isinstance(source,list) or not 1<=len(source)<=100:
        raise ValueError("Malformed original image-to-tasks source")
    result=[]
    for i,task in enumerate(source):
        if (not isinstance(task,dict) or task.get("image_id")!=key or
            task.get("domain")!=domain or type(task.get("ordered")) is not bool or
            not isinstance(task.get("intent"),str) or
            not 4<=len(task["intent"])<=1000):
            raise ValueError("Source task identity/domain/ordered/intent mismatch")
        intent=task["intent"]
        if any(s in intent.lower() for s in
             ("password","api key","social security","@gmail.","otp","private key")):
            raise ValueError("Exclude credential-related task instruction")
        centers=[]
        for box_key in ("start_bbox","end_bbox"):
            b=task.get(box_key)
            if (not isinstance(b,list) or len(b)!=4 or
                any(type(v) is not int for v in b)):
                raise ValueError("Original nonintegral or absent source box")
            x0,y0,x1,y1=b
            if not (0<=x0<x1<=width and 0<=y0<y1<=height):
                raise ValueError("Out-of-image original bounding box")
            centers.append([round((x0+x1)/2),round((y0+y1)/2)])
        if centers[0]==centers[1]:
            raise ValueError("Degenerate original drag")
        subtype=task.get("subtype")
        if subtype is not None and (not isinstance(subtype,str) or len(subtype)>140):
            raise ValueError("Invalid source subtype")
        result.append(dict(id=f"h57-{domain}-{key}-task{i:02d}",
            split="train",domain=domain,subtype=subtype,language="en",
            image_path=f"images/{domain}/{key}.jpg",image_sha256=sha(image),
            image_width=width,image_height=height,
            source_task_json_path=f"tasks/{domain}/{key}.json",
            source_task_sha256=sha(original_task_json),source_task_index=i,
            intent=intent,action_type="drag",ordered=task["ordered"],
            start_bbox=task["start_bbox"],end_bbox=task["end_bbox"],
            completion=dict(type="drag",start_point=centers[0],end_point=centers[1],
                coordinate_space="source_image_pixels",ordered=task["ordered"])))
    return result

def build(output):
    output=Path(output)
    if output.exists():raise FileExistsError("H57 original provider source is create-only")
    prepared=[]
    for domain in DOMAINS:
        image_range,ri=bounded_range(domain,"images")
        task_range,rt=bounded_range(domain,"tasks")
        images=tar_members(image_range,0,domain,"images")
        tasks=tar_members(task_range,DOMAINS[domain]["task_start"],domain,"tasks")
        matched=sorted(set(images)&set(tasks))
        if len(matched)<3:raise ValueError("Too few matching genuine TRAIN image/JSON pairs in "+domain)
        accepted=[];rejected=[]
        for key in matched:
            try:
                rows=verify_pair(domain,key,images[key]["payload"],tasks[key]["payload"])
                accepted.append((key,rows))
            except (ValueError,UnicodeError,TypeError) as exc:
                rejected.append(dict(source_image_id=key,reason=type(exc).__name__+":"+str(exc)[:120]))
        if len(accepted)<3:raise ValueError("No qualified image/JSON pair survived "+domain)
        print("H57_ORIGINAL_SOURCE_QUALIFIED",domain,
              "original_images",len(accepted),"native_actions",sum(len(r) for _,r in accepted),
              "source_rejected",len(rejected),flush=True)
        prepared.append((domain,images,tasks,accepted,rejected,ri,rt,
                         len(images),len(tasks),len(matched)))
    output.mkdir(parents=True)
    entries=[];rows=[];ranges=[];domain_summary={}
    for domain,images,tasks,accepted,rejected,ri,rt,raw_i,raw_t,nmatch in prepared:
        (output/"images"/domain).mkdir(parents=True)
        (output/"tasks"/domain).mkdir(parents=True)
        ranges.extend([ri,rt])
        for key,source_rows in accepted:
            a,b=images[key],tasks[key]
            (output/"images"/domain/(key+".jpg")).write_bytes(a["payload"])
            (output/"tasks"/domain/(key+".json")).write_bytes(b["payload"])
            rows.extend(source_rows)
            entries.append(dict(domain=domain,image_id=key,
               image_sha256=a["sha256"],task_sha256=b["sha256"],
               original_image_source_offset=a["offset"],original_image_size=a["size"],
               original_task_source_offset=b["offset"],original_task_size=b["size"],
               task_count=len(source_rows)))
        domain_summary[domain]=dict(accepted_source_images=len(accepted),
            native_drag_examples=sum(len(r) for _,r in accepted),
            original_image_headers_in_window=raw_i,
            original_task_headers_in_window=raw_t,
            matched_original_file_ids=nmatch,
            rejected_matched_pairs=rejected,
            ordered_true=sum(sum(r["ordered"] for r in rr) for _,rr in accepted),
            ordered_false=sum(sum(not r["ordered"] for r in rr) for _,rr in accepted))
    payload=b"".join(json.dumps(r,sort_keys=True,ensure_ascii=False,
                              separators=(",",":")).encode("utf8")+b"\n" for r in rows)
    (output/"train_drag_multidomain.jsonl").write_bytes(payload)
    manifest=dict(schema=SCHEMA,original_source_repo=SOURCE,original_revision=REV,
       source_license_label="Apache-2.0",full_upstream_archives_sha256_verified=False,
       only_original_train=True,benchmark_eval_or_test_used=False,
       model_weights_trained=False,privacy_review_complete=False,
       private_kaggle_upload_verified=False,domains=domain_summary,
       original_range_receipts=ranges,original_paired_image_task_members=entries,
       original_distinct_train_images=len(entries),original_train_drag_actions=len(rows),
       ordered_true=sum(r["ordered"] for r in rows),
       ordered_false=sum(not r["ordered"] for r in rows),
       training_label_sha256=sha(payload))
    (output/"manifest.json").write_bytes(canonical(manifest))
    audit(output)
    return dict(original_source_images=len(entries),native_train_drag_actions=len(rows),
                domains=domain_summary,source_jsonl_sha256=manifest["training_label_sha256"],
                real_training_executed=False,private_kaggle_uploaded=False)

def audit(root):
    root=Path(root)
    m=json.loads((root/"manifest.json").read_text("utf8"))
    if (m.get("schema")!=SCHEMA or m.get("original_revision")!=REV or
        m.get("only_original_train") is not True or
        m.get("benchmark_eval_or_test_used") is not False or
        m.get("model_weights_trained") is not False):
        raise ValueError("Source manifest not approved TRAIN-only identity")
    blob=(root/"train_drag_multidomain.jsonl").read_bytes()
    if sha(blob)!=m["training_label_sha256"]:
        raise ValueError("Original immutable training records changed")
    observed=[]
    for entry in m["original_paired_image_task_members"]:
        domain,key=entry["domain"],entry["image_id"]
        p=(root/"images"/domain/(key+".jpg")).read_bytes()
        t=(root/"tasks"/domain/(key+".json")).read_bytes()
        if sha(p)!=entry["image_sha256"] or sha(t)!=entry["task_sha256"]:
            raise ValueError("Original genuine image or source task bytes changed")
        rows=verify_pair(domain,key,p,t)
        if len(rows)!=entry["task_count"]:raise ValueError("Original number of tasks changed")
        observed.extend(rows)
    expected=b"".join(json.dumps(r,sort_keys=True,ensure_ascii=False,
                       separators=(",",":")).encode("utf8")+b"\n" for r in observed)
    if (blob!=expected or len(observed)!=m["original_train_drag_actions"] or
        len(m["original_paired_image_task_members"])!=m["original_distinct_train_images"] or
        sum(x["ordered"] for x in observed)!=m["ordered_true"] or
        sum(not x["ordered"] for x in observed)!=m["ordered_false"]):
        raise ValueError("Original source-to-training pairing/action semantics altered")
    return dict(screens=m["original_distinct_train_images"],
                actions=m["original_train_drag_actions"],
                ordered_true=m["ordered_true"],ordered_false=m["ordered_false"],
                original_train_sha256=m["training_label_sha256"])

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["fetch","audit"])
    parser.add_argument("--source",required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source) if args.command=="fetch" else audit(args.source),
          sort_keys=True))
