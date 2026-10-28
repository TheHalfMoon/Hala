"""H172 new original DEV GPU inputs must never include owner reference actions."""
import hashlib,json,zipfile
import pytest
import h172.infer_gpu as model
from h172.source_dev import cases

def make_zip(tmp_path,monkeypatch,mutation=None):
    archive=tmp_path/"h172-dev.pkg"
    rows=[];media={}
    for spec in cases():
        name="images/"+str(spec["index"])+".png"
        pixel=("fake-new-dev-only-"+str(spec["index"])).encode()
        rows.append({**spec,"image":name,
            "image_sha256":hashlib.sha256(pixel).hexdigest(),
            "instruction":"Select the original app item"})
        media[name]=pixel
    doc=dict(schema=model.DEV_SCHEMA,split="DEV",count=32,
             reference_gold_uploaded=False,
             previous_train_and_consumed_eval_excluded=True,
             private_cases=rows)
    if mutation:mutation(doc)
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("manifest.json",json.dumps(doc))
        for filename,raw in media.items():
            z.writestr(filename,raw)
    monkeypatch.setattr(model,"DEV_ZIP_SHA",model.sha_file(archive))
    return archive

def test_owner_gold_never_needed_for_admitted_gpu_source(tmp_path,monkeypatch):
    archive=make_zip(tmp_path,monkeypatch)
    rows=model.admit_dev(archive)
    assert len(rows)==32
    assert sum(r["language"]=="ar" for r in rows)==16
    assert all(r["split"]=="DEV" for r in rows)

@pytest.mark.parametrize("mutation",[
    lambda d:d.update(reference_gold_uploaded=True),
    lambda d:d.update(previous_train_and_consumed_eval_excluded=False),
    lambda d:d.update(split="TRAIN"),
    lambda d:d["private_cases"][0].update(training_supervision={"actions":[]}),
    lambda d:d["private_cases"][0].update(owner_reference_action=[1,2]),
    lambda d:d["private_cases"][0].update(image_sha256="bad"),
    lambda d:d["private_cases"][0].update(variant=100),
    lambda d:d["private_cases"][0].update(app="old-pikaday"),
    lambda d:d["private_cases"][0].update(language="xx")
])
def test_mutated_frozen_new_dev_refused(tmp_path,monkeypatch,mutation):
    archive=make_zip(tmp_path,monkeypatch,mutation)
    with pytest.raises(ValueError):
        model.admit_dev(archive)

def test_not_pinned_archive_refused(tmp_path,monkeypatch):
    archive=make_zip(tmp_path,monkeypatch)
    monkeypatch.setattr(model,"DEV_ZIP_SHA","0"*64)
    with pytest.raises(ValueError):
        model.admit_dev(archive)
