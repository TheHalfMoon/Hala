"""H167 independent syntax- and safety-level preflights; no real heldout media."""
import pytest
from h167.combined_probe import (
    check_adapter_partition,is_visual_trainable,is_language_adapter,
    language_rows,admit_frozen_language_zip)

class P:
    def __init__(self,train):self.requires_grad=train

def safe_params():
    return [
       ("base_model.model.model.language_model.layers.0.q_proj.lora_A.language.weight",P(False)),
       ("base_model.model.model.language_model.layers.0.q_proj.lora_B.language.weight",P(False)),
       ("base_model.model.model.visual.merger.linear_fc1.lora_A.vision.weight",P(True)),
       ("base_model.model.model.visual.merger.linear_fc1.lora_B.vision.weight",P(True)),
       ("base_model.model.model.visual.merger.linear_fc2.lora_A.vision.weight",P(True)),
       ("base_model.model.model.visual.merger.linear_fc2.lora_B.vision.weight",P(True)),
       ("base_model.model.model.visual.blocks.0.q_proj.weight",P(False))]

def test_exact_frozen_language_and_trainable_merger_partition():
    lang,visual=check_adapter_partition(safe_params())
    assert len(lang)==2 and len(visual)==4
    assert is_visual_trainable(visual[0][0])
    assert is_language_adapter(lang[0][0])

@pytest.mark.parametrize("index,replacement",[
    (0,True),(2,False),(5,False),(6,True)])
def test_misconfigured_adapter_partition_fails(index,replacement):
    params=safe_params()
    params[index][1].requires_grad=replacement
    with pytest.raises(ValueError):
        check_adapter_partition(params)

def test_missing_language_or_extra_visual_fails():
    params=safe_params()
    with pytest.raises(ValueError):
        check_adapter_partition(params[2:])
    with pytest.raises(ValueError):
        check_adapter_partition(params+[("model.visual.merger.linear_fc3.lora_A.vision.weight",P(True))])

def test_reject_wrong_h164_frozen_zip_without_extract(tmp_path):
    package=tmp_path/"forged.zip"
    package.write_bytes(b"not the frozen H164 archive")
    with pytest.raises(ValueError):
        admit_frozen_language_zip(package,tmp_path/"extracted")
    assert not (tmp_path/"extracted").exists()

def test_synthetic_bilingual_has_same_non_private_gold():
    pytest.importorskip("PIL")
    rows=language_rows()
    assert [r["language"] for _,r in rows]==["en","ar"]
    assert rows[0][1]["training_supervision"]==rows[1][1]["training_supervision"]
    assert rows[0][0].getpixel((480,360))==(255,0,0)
    assert "المربع" in rows[1][1]["instruction"]

class FakeLoRALayer:
    def __init__(self,adapters,active):
        self.lora_A={name:object() for name in adapters}
        self.active_adapters=list(active)

def actual_modules():
    from h167.combined_probe import verify_active_lora_modules
    modules=[
        ("base_model.model.model.language_model.layers.0.self_attn.q_proj",
         FakeLoRALayer(["language"],["language","vision"])),
        ("base_model.model.model.visual.merger.linear_fc1",
         FakeLoRALayer(["vision"],["language","vision"])),
        ("base_model.model.model.visual.merger.linear_fc2",
         FakeLoRALayer(["vision"],["language","vision"]))]
    return verify_active_lora_modules,modules

def test_h167_actual_lora_modules_need_language_and_both_visual_active():
    verify,modules=actual_modules()
    qualified=verify(modules)
    assert qualified["active_language_layers"]==1
    assert len(qualified["active_vision_modules"])==2

@pytest.mark.parametrize("index,active",[
    (0,["vision"]),(1,["language"]),(2,[])])
def test_h167_inactive_underlying_lora_module_rejected(index,active):
    verify,modules=actual_modules()
    modules[index][1].active_adapters=active
    with pytest.raises(ValueError):
        verify(modules)

def test_h167_missing_visual_layer_rejected():
    verify,modules=actual_modules()
    with pytest.raises(ValueError):
        verify(modules[:2])
