"""H139 frozen cross-repository source admissibility contract.

Never train on H136/H137/H138 evaluated screenshots or the independent
H139 blind evaluation app. All upstream MIT notices stay intact.
"""
from pathlib import Path
import hashlib
import subprocess

SCHEMA="hala.h139.legal-MIT-cross-app-source-splits.v1"
UPSTREAM={
 "js-board":{
  "git_sha":"82325355d168a9f4dba9f683899a0a6991034111",
  "role":"train","engine":"vanilla-modular-HTML5","license_owner":"CodeWithSindhu",
  "built_dir":"dist"},
 "alt-static-board":{
  "git_sha":"a4426599cb3778c8ee3d7c6b1fd27eed76bf5fe9",
  "role":"train","engine":"vanilla-HTML5","license_owner":"Arjun Vashishtha",
  "built_dir":"."},
 "ood-board":{
  "git_sha":"7ce11ad463b01a7b12b105417453d5251eabcd51",
  "role":"blind_test","engine":"react-dnd-HTML5","license_owner":"Gionatha Sturba",
  "built_dir":"dist"},
}

def sha(blob):return hashlib.sha256(blob).hexdigest()

def check_upstream(folder,name):
    folder=Path(folder)
    if name not in UPSTREAM:raise ValueError("Unknown H139 original application")
    p=UPSTREAM[name]
    actual=subprocess.check_output(
        ["git","-C",str(folder),"rev-parse","HEAD"],text=True).strip()
    if actual!=p["git_sha"]:raise ValueError("H139 immutable source SHA drift")
    modified=subprocess.check_output(
        ["git","-C",str(folder),"diff","--name-only","HEAD"],text=True).strip()
    if modified:raise ValueError("H139 upstream tracked source modified")
    lic=folder/"LICENSE"
    if not lic.is_file():raise ValueError("H139 explicit source MIT LICENSE required")
    raw=lic.read_text(encoding="utf8")
    if not raw.startswith("MIT License") or "Permission is hereby granted" not in raw:
        raise ValueError("H139 incomplete MIT license")
    if p["license_owner"].lower() not in raw.lower():
        raise ValueError("H139 original author attribution missing")
    if not (folder/p["built_dir"]/"index.html").is_file():
        raise ValueError("H139 independent offline app build missing")
    return {"app":name,"git_sha":actual,"license_sha256":sha(lic.read_bytes()),
            "role":p["role"],"engine":p["engine"]}

def check_split_manifest(manifest,forbidden_sha=()):
    if manifest.get("schema")!=SCHEMA or manifest.get("private") is not True:
        raise ValueError("H139 private provenance not sealed")
    rows=manifest.get("cases")
    if not isinstance(rows,list) or not rows:
        raise ValueError("H139 source case list missing")
    old=set(forbidden_sha)
    seen=set()
    for row in rows:
        app=row.get("app")
        if app not in UPSTREAM:raise ValueError("H139 source app not admitted")
        policy=UPSTREAM[app]
        if (row.get("role")!=policy["role"] or
            row.get("app_git_sha")!=policy["git_sha"]):
            raise ValueError("H139 app role/immutable SHA mismatch")
        h=row.get("image_sha256")
        if not isinstance(h,str) or len(h)!=64 or h in seen:
            raise ValueError("H139 image hash duplicate or malformed")
        seen.add(h)
        if row.get("role")=="train" and (app=="ood-board" or h in old):
            raise ValueError("H139 blind-test or previous heldout leaked to TRAIN")
    return rows
