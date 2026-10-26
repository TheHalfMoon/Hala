"""H136 private ZIP transport admission — original images only.

Work around multi-file Kaggle upload failures by transferring frozen owner
corpus as one private ZIP per split. Extract only exact SHA-qualified PNGs
and manifest into scratch; NEVER mount a sealed split in training.
"""
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from h136.source import SCHEMA,check,SPLITS

FILES={"train":"train288.zip","dev":"dev24.zip"}
MAX_ZIP_UNPACK_BYTES=16_000_000

def admit_zipped_private_sources(kaggle_input,root):
    kaggle_input=Path(kaggle_input)
    root=Path(root)
    if not kaggle_input.is_dir() or root.exists():
        raise ValueError("H136 private source root absent or scratch already used")
    located={}
    for split,filename in FILES.items():
        matches=list(kaggle_input.rglob(filename))
        if len(matches)!=1:
            raise ValueError("H136 exactly one owner-private "+filename+" must be mounted")
        located[split]=matches[0]
    if any("sealed" in p.name.lower() for p in kaggle_input.rglob("*.zip")):
        raise ValueError("H136 SEALED split must not be attached to optimizer kernel")
    for split,path in located.items():
        with ZipFile(path,"r") as archive:
            infos=archive.infolist()
            if len(infos)!=SPLITS[split][1]+1:
                raise ValueError("H136 unexpected archive row count")
            names=[x.filename for x in infos]
            if (len(names)!=len(set(names)) or
                any("/" in name or "\\" in name or ".." in name or not name
                    for name in names)):
                raise ValueError("H136 ZIP path traversal or duplicate names rejected")
            if "manifest.json" not in names:
                raise ValueError("H136 SHA-qualified original manifest missing")
            manifest=json.loads(archive.read("manifest.json").decode("utf8"))
            if manifest.get("schema")!=SCHEMA or manifest.get("split")!=split:
                raise ValueError("H136 original transport split/schema mismatch")
            expected={"manifest.json"}|{c["image"] for c in manifest.get("cases",[])}
            if set(names)!=expected:
                raise ValueError("H136 ZIP files not exact frozen source list")
            if sum(i.file_size for i in infos)>MAX_ZIP_UNPACK_BYTES:
                raise ValueError("H136 source private ZIP uncompressed bytes over budget")
            folder=root/split
            folder.mkdir(parents=True,exist_ok=False)
            for info in infos:
                (folder/info.filename).write_bytes(archive.read(info.filename))
            check(folder,split)
    train,dev=check(root/"train","train"),check(root/"dev","dev")
    if {c["image_sha256"] for c in train["cases"]}&{
            c["image_sha256"] for c in dev["cases"]}:
        raise ValueError("H136 ZIP original train/dev image SHA overlap")
    return root
