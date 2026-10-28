"""H166: synthetic-only low-VRAM vision projection targeting guards."""
import pytest
from h166.vision_probe import (
    visual_merger_targets,require_free_t4,synth,
    EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT)

class Linear:pass
class Other:pass

def test_targets_only_two_visual_merger_linear_modules():
    names=[
      ("model.visual.merger.mlp.0",Linear()),
      ("model.visual.merger.mlp.2",Linear()),
      ("model.language_model.layers.0.mlp.0",Linear()),
      ("model.visual.blocks.0.attn.q_proj",Linear()),
      ("model.visual.merger.act",Other())]
    assert visual_merger_targets(names,Linear)==[
        "model.visual.merger.mlp.0","model.visual.merger.mlp.2"]

@pytest.mark.parametrize("names",[
    [],[("model.language_model.layers.1.merger.visual",Linear())],
    [("model.visual.blocks.0.mlp.0",Linear())],
    [("model.visual.merger.mlp.0",Other())],
    [(f"model.visual.merger.mlp.{i}",Linear()) for i in range(5)]])
def test_missing_or_unsafe_targets_fail_closed(names):
    with pytest.raises(ValueError):
        visual_merger_targets(names,Linear)

class FakeCuda:
    def __init__(self,names):self.names=names
    def is_available(self):return bool(self.names)
    def device_count(self):return len(self.names)
    def get_device_name(self,i):return self.names[i]
class FakeTorch:
    def __init__(self,names):self.cuda=FakeCuda(names)
ENV={"torch":EXPECTED_TORCH,"transformers":EXPECTED_TRANSFORMERS,
     "peft":EXPECTED_PEFT,"Pillow":"11.3.0"}

def test_exact_free_cuda_t4_only():
    assert require_free_t4(FakeTorch(["Tesla T4","Tesla T4"]),ENV)=="Tesla T4"
    for names in ([],["A100"],["Tesla T4","A100"]):
        with pytest.raises(ValueError):
            require_free_t4(FakeTorch(names),ENV)
    with pytest.raises(ValueError):
        require_free_t4(FakeTorch(["Tesla T4"]),dict(ENV,peft="0.0.0"))

def test_synthetic_only_image_has_no_original_app_pixels():
    pytest.importorskip("PIL")
    img,row=synth()
    assert img.size==(960,720)
    assert img.getpixel((480,360))==(255,0,0)
    assert img.getpixel((0,0))==(255,255,255)
    assert row["training_supervision"]["actions"][0]["point"]==[480,360]
