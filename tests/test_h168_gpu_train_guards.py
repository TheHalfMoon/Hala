"""H168 no-OOD real source training structural and private-checkpoint safety tests."""
import hashlib
import io
import json
import struct
import zipfile

import pytest
from h168 import train_gpu
from h168.audit_gpu import safely_read_vision_only_weights

def sha(x):
    return hashlib.sha256(x).hexdigest()

def fake_package(path,*,mode="good"):
    rows=[]
    images={}
    for app,base,kind in (("choices",168000,"type"),("nouislider",168100,"drag")):
        for i in range(16):
            index=base+i
            name="images/"+str(index)+".png"
            blob=("fake-only-test-image-"+str(index)).encode()
            gold={"kind":kind,"point":[40,50],"text":"tag"} if kind=="type" else {
                "kind":"drag","start":[40,50],"end":[70,50]}
            row=dict(split="TRAIN",app=app,index=index,image=name,
                     image_sha256=sha(blob),language="en" if i%2==0 else "ar",
                     action_type=kind,instruction="test only original action",
                     training_supervision={"actions":[gold]})
            rows.append(row);images[name]=blob
    doc=dict(schema=train_gpu.TRAIN_SCHEMA,split="TRAIN",count=32,
             dev_ood_count=0,reference_gold_uploaded_for_train_only=True,
             historical_consumed_images_excluded=True,
             data_role="H168_NEW_SOURCE_TRAIN_PILOT_ONLY",
             cases=rows)
    if mode=="leak_dev":
        doc["dev_ood_count"]=1
    if mode=="duplicate":
        doc["cases"][1]["image_sha256"]=doc["cases"][0]["image_sha256"]
    if mode=="bad_label":
        doc["cases"][0]["training_supervision"]["actions"][0]["kind"]="drag"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(doc))
        for name,blob in images.items():
            z.writestr(name,blob)
        if mode=="extra":
            z.writestr("private-ood-gold.json","forbidden")
    return sha(path.read_bytes())

def test_exact_32_train_package_structure(monkeypatch,tmp_path):
    path=tmp_path/"train.pkg"
    monkeypatch.setattr(train_gpu,"TRAIN_ZIP_SHA",fake_package(path))
    rows=train_gpu.admit_train(path)
    assert len(rows)==32
    assert sum(r["language"]=="ar" for r in rows)==16
    assert sum(r["action_type"]=="drag" for r in rows)==16

@pytest.mark.parametrize("mode",["leak_dev","duplicate","bad_label","extra"])
def test_invalid_source_or_gold_fails_closed(monkeypatch,tmp_path,mode):
    path=tmp_path/"train.pkg"
    monkeypatch.setattr(train_gpu,"TRAIN_ZIP_SHA",fake_package(path,mode=mode))
    with pytest.raises(ValueError):
        train_gpu.admit_train(path)

def test_unpinned_train_package_rejected(tmp_path):
    path=tmp_path/"train.pkg"
    fake_package(path)
    with pytest.raises(ValueError):
        train_gpu.admit_train(path)

def fake_safetensors(keys):
    header={name:{"dtype":"F16","shape":[1],"data_offsets":[i*2,i*2+2]}
            for i,name in enumerate(keys)}
    raw=json.dumps(header).encode()
    return struct.pack("<Q",len(raw))+raw+b"\x00"*(2048-len(raw)-8)

def test_private_vision_checkpoint_rejects_language_tensor(tmp_path):
    safe=[
        "base_model.model.model.visual.merger.linear_fc1.lora_A.weight",
        "base_model.model.model.visual.merger.linear_fc1.lora_B.weight",
        "base_model.model.model.visual.merger.linear_fc2.lora_A.weight",
        "base_model.model.model.visual.merger.linear_fc2.lora_B.weight"]
    blob=fake_safetensors(safe)
    path=tmp_path/"adapter_model.safetensors"
    path.write_bytes(blob)
    assert safely_read_vision_only_weights(path)==sha(blob)
    safe[0]="base_model.model.model.language_model.layers.1.lora_A.weight"
    path.write_bytes(fake_safetensors(safe))
    with pytest.raises(ValueError):
        safely_read_vision_only_weights(path)
