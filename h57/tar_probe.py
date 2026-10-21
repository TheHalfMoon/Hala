"""Read-only HTTP206 bounded metadata probes for three official DragOn TRAIN TARs.

No full TAR downloaded. Never accesses eval/test. Probes are not imported data.
"""
from __future__ import annotations
import argparse,json,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
REV="3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
TRAIN_BYTES={"sheet":14258661376,"slide_resize":9915637248,"text_highlight":29874024960}
N=1048576
def headers(data,start,name):
    assert len(data)==N and start%512==0
    out=[]
    for i in range(0,N-512,512):
        b=data[i:i+512]
        try:
            if b[257:262]!=b"ustar":continue
            if sum(b[:148])+8*32+sum(b[156:])!=int(b[148:156].strip(b" \0") or b"0",8):continue
            s=b[:100].split(b"\0")[0].decode("ascii")
            n=int(b[124:136].strip(b" \0") or b"0",8)
            if (not s.startswith(name+"/") or "/._" in s or "/PaxHeader/" in s
                or not s.endswith((".jpg",".json"))):continue
            out.append(dict(name=s,size=n,offset=start+i+512,
                            kind="tasks" if "/tasks/" in s else "images"))
        except (ValueError,UnicodeError):pass
    return out
def fetch(name,start):
    total=TRAIN_BYTES[name]
    start=(start//512)*512
    if start<0 or start+N>total:raise ValueError("Range outside original TRAIN TAR")
    url=f"https://huggingface.co/datasets/Hcompany/DragOn/resolve/{REV}/train/{name}.tar?download=true"
    req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{start+N-1}",
             "User-Agent":"Hala-official-DragOn-TRAIN-metadata-2026-10-21"})
    with urllib.request.urlopen(req,timeout=45) as r:
        if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{start+N-1}/{total}":
            raise ValueError("Provider range/content identity incorrect")
        data=r.read(N+1)
    if len(data)!=N:raise ValueError("Provider gave incomplete/unbounded TAR bytes")
    return headers(data,start,name)
def locate(name):
    total=TRAIN_BYTES[name]
    lo,hi=0,((total-N)//512)*512
    points=[]
    for i in range(21):
        mid=(((lo+hi)//2)//512)*512
        found=fetch(name,mid)
        if not found:
            # In a large JPEG within 1MiB; attempt adjacent offset (bounded) before failing
            alt=min(hi,mid+N)
            found=fetch(name,alt)
            if not found:raise ValueError("No real TAR header in two adjacent verified ranges")
        kinds={t["kind"] for t in found}
        anchor="tasks" if "tasks" in kinds else "images"
        points.append(dict(start=mid,anchor=anchor,example=found[0]["name"]))
        if anchor=="tasks":hi=mid
        else:lo=mid
        print("H57_BOUNDARY_PROBE",name,i,"approx",mid,"part",anchor,flush=True)
        if hi-lo<=2*N:break
    marker=((hi+N)//N)*N
    return dict(domain=name,source_repo="Hcompany/DragOn",source_rev=REV,split="train",
                tar_total_bytes=total,boundary_probe_before=lo,boundary_probe_after=hi,
                candidate_tasks_start=marker,points=points,
                full_tar_content_sha256_independently_verified=False,
                training_data_downloaded=False,models_trained=False)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--domain",choices=list(TRAIN_BYTES)+["all"],default="all")
    args=parser.parse_args()
    if args.domain=="all":
        with ThreadPoolExecutor(max_workers=3) as pool:
            future={pool.submit(locate,d):d for d in TRAIN_BYTES}
            result={future[f]:f.result() for f in as_completed(future)}
    else:result={args.domain:locate(args.domain)}
    print("H57_BOUNDARY_RESULT",json.dumps(result,sort_keys=True),flush=True)
