"""H53: authentic pinned H Company DragOn TRAIN-only slider image/task shard.

Reads bounded HTTP 206 TAR *byte ranges*, joins genuine original JPEG/JSON
by exact UUID (not TAR order), retains original JPEG and task JSON bytes,
and emits reversible source records with an explicit drag action type.
Never accesses eval/test, runs a model, downloads 7GB, or uses paid services.
"""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import urllib.request
from PIL import Image

SOURCE_REPO="Hcompany/DragOn"
SOURCE_REV="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
SOURCE_FILE="train/slider.tar"
SOURCE_TOTAL=7146830848
SOURCE_POINTER_SHA="2455d5b4a164d4ff3d6cf41d92d24ad2e27db29e28ea16f5585b55956356a33e"
URL=f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REV}/{SOURCE_FILE}?download=true"
IMAGE_START=0
TASK_START=6231427584  # original 512-byte-aligned TAR block near start of TRAIN task members
WINDOW_BYTES=64*1024*1024
SCHEMA="hala.h53.dragon-authentic-slider-train.v1"
UUID=re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(o):return (json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode("utf-8")

def fetch(start,span):
    if type(start) is not int or type(span) is not int or span!=WINDOW_BYTES or start not in (IMAGE_START,TASK_START):
        raise ValueError("Only frozen bounded original TRAIN TAR windows permitted")
    req=urllib.request.Request(URL,headers={
      "Range":f"bytes={start}-{start+span-1}",
      "User-Agent":"Hala-H53-frozen-train-range-client/1"})
    with urllib.request.urlopen(req,timeout=120) as response:
        expected_range=f"bytes {start}-{start+span-1}/{SOURCE_TOTAL}"
        if (response.status!=206 or response.headers.get("Content-Range")!=expected_range or
            response.headers.get("Content-Length")!=str(span)):
            raise ValueError("Provider did not furnish precisely bounded immutable TRAIN content")
        data=response.read(span+1)
        if len(data)!=span:raise ValueError("Incomplete/unbounded upstream range")
    return data,dict(offset=start,length=len(data),sha256=sha(data),
                     status=206,content_range=expected_range)

def tar_members(data,base,kind):
    """Scan valid original TAR headers, not arbitrary JPG/JSON byte signatures."""
    if len(data)!=WINDOW_BYTES or base not in (IMAGE_START,TASK_START):
        raise ValueError("Unqualified TAR range input")
    found={}
    extension=".jpg" if kind=="images" else ".json"
    expected_prefix=f"slider/{kind}/"
    for offset in range(0,len(data)-512,512):
        block=data[offset:offset+512]
        if block[257:262] not in (b"ustar",b"\x00\x00\x00\x00\x00"):
            continue
        try:
            value=int(block[148:156].strip(b" \0") or b"0",8)
            if sum(block[:148])+8*32+sum(block[156:])!=value:continue
            name=block[:100].split(b"\0")[0].decode("ascii")
            n=int(block[124:136].strip(b" \0") or b"0",8)
            if (not name.startswith(expected_prefix) or not name.endswith(extension) or
                "/._" in name or "/PaxHeader/" in name or
                block[156:157] not in (b"0",b"\0") or n<=0 or n>2*1024*1024 or
                offset+512+n>len(data)):
                continue
            item=name[len(expected_prefix):-len(extension)]
            if not UUID.fullmatch(item):continue
            if item in found:raise ValueError("Duplicate original member ID in TAR")
            payload=data[offset+512:offset+512+n]
            found[item]=dict(name=name,bytes=payload,sha256=sha(payload),
                             offset=base+offset+512,size=n)
        except (UnicodeDecodeError,ValueError) as error:
            if "Duplicate original member" in str(error):raise
            continue
    return found

def verify_item(key,photo,raw):
    if not UUID.fullmatch(key):raise ValueError("Invalid original source ID")
    if not (photo[:3]==b"\xff\xd8\xff" and photo[-2:]==b"\xff\xd9"):
        raise ValueError("Original file not genuine complete JPEG")
    with Image.open(BytesIO(photo)) as im:
        if im.format!="JPEG":raise ValueError("Image encoding mismatch")
        width,height=im.size
        im.verify()
    if width<300 or height<300 or width*height>5_000_000:
        raise ValueError("Unexpected original screenshot size")
    original=json.loads(raw.decode("utf-8"))
    if not isinstance(original,list) or not 1<=len(original)<=40:
        raise ValueError("Original DragOn slider multi-task JSON")
    rows=[]
    for index,task in enumerate(original):
        if not isinstance(task,dict) or any(k not in task for k in
          ("image_id","intent","start_bbox","end_bbox","ordered","domain")):
            raise ValueError("Missing original task identity/coordinates")
        if (task["image_id"]!=key or task["domain"]!="slider" or task["ordered"] is not True or
            not isinstance(task["intent"],str) or not 3<=len(task["intent"])<=500 or
            any(x in task["intent"].lower() for x in ("password", "api key", "social security", "@gmail."))):
            raise ValueError("Original slider task identity/intent restriction")
        centers=[]
        for field in ("start_bbox","end_bbox"):
            box=task[field]
            if (not isinstance(box,list) or len(box)!=4 or
                not all(type(v) is int for v in box)):
                raise ValueError("Invalid original bounding box")
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=width and 0<=y0<y1<=height):
                raise ValueError("DragOn box outside original JPEG")
            centers.append([round((x0+x1)/2),round((y0+y1)/2)])
        if centers[0]==centers[1]:
            raise ValueError("Zero-length slider drag")
        rows.append(dict(id=f"h53-{key}-task{index:02d}",split="train",domain="slider",
            action_type="drag",language="en",intent=task["intent"],
            prompt=task["intent"],image_path=f"images/{key}.jpg",image_sha256=sha(photo),
            source_task_sha256=sha(raw),source_task_index=index,
            image_width=width,image_height=height,ordered=True,
            start_bbox=task["start_bbox"],end_bbox=task["end_bbox"],
            completion=dict(type="drag",start_point=centers[0],
             end_point=centers[1],coordinate_space="source_image_pixels",ordered=True)))
    return rows

def build(folder):
    folder=Path(folder)
    if folder.exists():raise FileExistsError("H53 first original pinned shard must be create-only")
    images_data,receipt1=fetch(IMAGE_START,WINDOW_BYTES)
    tasks_data,receipt2=fetch(TASK_START,WINDOW_BYTES)
    imgs=tar_members(images_data,IMAGE_START,"images")
    tasks=tar_members(tasks_data,TASK_START,"tasks")
    matches=sorted(set(imgs)&set(tasks))
    if len(matches)<5:raise ValueError("Not enough genuinely matching TRAIN JPEG+JSON IDs")
    accepted=[];all_rows=[];rejected=[]
    for key in matches:
        try: rows=verify_item(key,imgs[key]["bytes"],tasks[key]["bytes"])
        except (ValueError,UnicodeError,TypeError) as exc:
            rejected.append(dict(image_id=key,reason=type(exc).__name__+":"+str(exc)[:150]))
            continue
        accepted.append((key,rows))
        all_rows.extend(rows)
    if len(accepted)<5 or len(all_rows)<15:raise ValueError("No sufficiently qualified authentic slider shard")
    folder.mkdir(parents=True)
    (folder/"images").mkdir()
    (folder/"tasks").mkdir()
    entries=[]
    for key,rows in accepted:
        image=imgs[key]
        task=tasks[key]
        (folder/"images"/(key+".jpg")).write_bytes(image["bytes"])
        (folder/"tasks"/(key+".json")).write_bytes(task["bytes"])
        entries.append(dict(image_id=key,source_image_range=[image["offset"],image["size"]],
          source_task_range=[task["offset"],task["size"]],
          image_sha256=image["sha256"],task_sha256=task["sha256"],
          examples=len(rows)))
    authoring=b"".join(json.dumps(row,sort_keys=True,ensure_ascii=False).encode("utf-8")+b"\n" for row in all_rows)
    (folder/"train_drag.jsonl").write_bytes(authoring)
    proof=dict(schema=SCHEMA,source_dataset=SOURCE_REPO,source_revision=SOURCE_REV,
      source_file=SOURCE_FILE,provider_declared_tar_bytes=SOURCE_TOTAL,
      provider_lfs_pointer_sha256=SOURCE_POINTER_SHA,
      full_source_tar_sha256_independently_verified=False,
      source_license_label="Apache-2.0",
      source_train_only=True,source_eval_used=False,
      public_benchmark_answers_used=False,
      real_model_training_executed=False,remote_storage_verified=False,
      visual_privacy_review_independently_complete=False,
      train_rows=len(all_rows),unique_original_screenshots=len(entries),
      range_receipts=[receipt1,receipt2],matched_source_entries=entries,
      rejected_source_entries=rejected,
      unmatched_image_count=len(imgs)-len(matches),
      unmatched_task_count=len(tasks)-len(matches),
      source_images_in_range=len(imgs),source_tasks_in_range=len(tasks),
      jsonl_sha256=sha(authoring))
    (folder/"manifest.json").write_bytes(canonical(proof))
    return audit(folder)

def audit(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text("utf-8"))
    if (manifest.get("schema")!=SCHEMA or manifest["source_revision"]!=SOURCE_REV or
        manifest["source_file"]!=SOURCE_FILE or manifest["source_train_only"] is not True or
        manifest["source_eval_used"] is not False or
        manifest["real_model_training_executed"] is not False or
        manifest["full_source_tar_sha256_independently_verified"] is not False):
        raise ValueError("Unqualified H53 original source")
    rows=[json.loads(line) for line in (folder/"train_drag.jsonl").read_text("utf-8").splitlines()]
    if manifest["jsonl_sha256"]!=sha((folder/"train_drag.jsonl").read_bytes()):
        raise ValueError("Original H53 rows changed")
    rebuilt=[];ids=set()
    for entry in manifest["matched_source_entries"]:
        key=entry["image_id"]
        if key in ids or not UUID.fullmatch(key):raise ValueError("Duplicate image")
        ids.add(key)
        image=(folder/"images"/(key+".jpg")).read_bytes()
        task=(folder/"tasks"/(key+".json")).read_bytes()
        if sha(image)!=entry["image_sha256"] or sha(task)!=entry["task_sha256"]:
            raise ValueError("Original source member bytes changed")
        rebuilt.extend(verify_item(key,image,task))
    if (rows!=rebuilt or len(rows)!=manifest["train_rows"] or
        len(ids)!=manifest["unique_original_screenshots"] or
        len(ids)<5 or len(rows)<15):
        raise ValueError("Original source-to-supervision bridge changed")
    return dict(original_train_images=len(ids),original_train_drag_annotations=len(rows),
       source_jsonl_sha256=manifest["jsonl_sha256"],
       source_repo=SOURCE_REPO,source_revision=SOURCE_REV,
       model_trained=False,remote_stored=manifest["remote_storage_verified"])

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["fetch","audit"])
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(build(args.output) if args.command=="fetch" else audit(args.output),
                     sort_keys=True))
