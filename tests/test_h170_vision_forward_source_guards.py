"""H170 diagnostic is TRAIN-only, never a disguised DEV/OOD re-evaluation."""
import hashlib
import json
import zipfile
import pytest
from h170 import sensitivity_gpu as diag

def fixture(tmp_path,monkeypatch):
    row=dict(split="TRAIN",index=168000,app="choices",
             language="en",image="images/168000.png",
             image_sha256=hashlib.sha256(b"TRAIN-ONLY").hexdigest(),
             instruction="Test training UI only",
             training_supervision={"actions":[{"kind":"type","point":[20,20],
                                               "text":"train"}]})
    archive=tmp_path/"train.pkg"
    names=[168000+i for i in range(16)]+[168100+i for i in range(16)]
    doc=dict(schema="hala.h168.owner-private-new-original-train-32.v1",
             split="TRAIN",dev_ood_count=0,cases=[row]+[
                 dict(row,index=i,image="images/"+str(i)+".png")
                 for i in names if i!=168000])
    def write():
        with zipfile.ZipFile(archive,"w") as z:
            z.writestr("manifest.json",json.dumps(doc))
            for ix in names:
                z.writestr("images/"+str(ix)+".png",
                           b"TRAIN-ONLY" if ix==168000 else str(ix).encode())
        monkeypatch.setattr(diag,"TRAIN_ZIP_SHA",
            hashlib.sha256(archive.read_bytes()).hexdigest())
    write()
    return doc,archive,write

def test_train_diagnostic_prompt_omits_gold(tmp_path,monkeypatch):
    doc,path,write=fixture(tmp_path,monkeypatch)
    pixels,row=diag.one_train_image(path)
    assert pixels==b"TRAIN-ONLY"
    assert row["split"]=="TRAIN_DIAGNOSTIC_INPUT_ONLY"
    assert "training_supervision" not in row

@pytest.mark.parametrize("mutator",[
    lambda d:d.update(split="DEV"),
    lambda d:d.update(dev_ood_count=1),
    lambda d:d["cases"][0].update(index=999),
    lambda d:d["cases"][0].update(image_sha256="corrupted"),
    lambda d:d["cases"][0].update(app="pikaday"),
])
def test_old_dev_or_mismatched_train_rejected(tmp_path,monkeypatch,mutator):
    doc,path,write=fixture(tmp_path,monkeypatch)
    mutator(doc);write()
    with pytest.raises(ValueError):
        diag.one_train_image(path)

def test_wrong_package_sha_fails(tmp_path,monkeypatch):
    doc,path,write=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(diag,"TRAIN_ZIP_SHA","0"*64)
    with pytest.raises(ValueError):
        diag.one_train_image(path)

def test_paired_tensor_delta_not_inferred_from_greedy_text():
    torch=pytest.importorskip("torch")
    a=torch.tensor([1.0,2.0])
    b=torch.tensor([1.0,2.001])
    m=diag.absolute_norm_metrics(a,b)
    assert m["nonzero_elements"]==1
    assert m["max_abs_delta"]>0
    assert diag.absolute_norm_metrics(a,a)["nonzero_elements"]==0
