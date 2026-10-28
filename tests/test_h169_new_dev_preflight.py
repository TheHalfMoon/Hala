"""H169 new DEV is disjoint and GPU inputs fail closed on owner-gold leakage."""
import json
import pytest
from h169.source_dev import cases,SCHEMA
from h169.infer_gpu import admit_dev,sha_file

def fixture(tmp_path,monkeypatch):
    import h169.infer_gpu as gpu
    import zipfile,hashlib
    rows=[]
    files={}
    for c in cases():
        pic=("isolated fake dev original-"+str(c["index"])).encode()
        name="images/"+str(c["index"])+".png"
        rows.append({**c,"image":name,"instruction":"Click/type/drag visible task",
                    "image_sha256":hashlib.sha256(pic).hexdigest()})
        files[name]=pic
    doc=dict(schema=gpu.DEV_SCHEMA,split="DEV",count=32,
             reference_gold_uploaded=False,historical_train_ood_excluded=True,
             private_cases=rows)
    path=tmp_path/"dev.pkg"
    def write():
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("manifest.json",json.dumps(doc))
            for name,pic in files.items():z.writestr(name,pic)
        monkeypatch.setattr(gpu,"DEV_ZIP_SHA",sha_file(path))
    write()
    return doc,path,write

def test_preregistered_32_dev_splits_and_languages():
    plan=cases()
    assert len(plan)==32
    assert len({x["index"] for x in plan})==32
    assert {x["variant"] for x in plan}==set(range(16,32))
    assert sum(x["language"]=="ar" for x in plan)==16
    assert {x["app"] for x in plan}=={"choices","nouislider"}

def test_dev_only_zip_preflight_passes_with_no_gold(tmp_path,monkeypatch):
    doc,path,write=fixture(tmp_path,monkeypatch)
    assert len(admit_dev(path))==32

@pytest.mark.parametrize("change",[
    lambda x:x.update(reference_gold_uploaded=True),
    lambda x:x.update(historical_train_ood_excluded=False),
    lambda x:x.update(split="TRAIN"),
    lambda x:x["private_cases"][0].update(training_supervision={"actions":[]}),
    lambda x:x["private_cases"][0].update(image_sha256="bad"),
    lambda x:x["private_cases"][0].update(variant=0),
    lambda x:x["private_cases"][0].update(language="zz"),
    lambda x:x["private_cases"][0].update(reference_point=[15,60]),
])
def test_dev_manifest_mutations_fail_closed(tmp_path,monkeypatch,change):
    doc,path,write=fixture(tmp_path,monkeypatch)
    change(doc);write()
    with pytest.raises(ValueError):
        admit_dev(path)

def test_pinned_dev_source_hash_prevents_mutation(tmp_path,monkeypatch):
    doc,path,write=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr("h169.infer_gpu.DEV_ZIP_SHA","a"*64)
    with pytest.raises(ValueError):
        admit_dev(path)
