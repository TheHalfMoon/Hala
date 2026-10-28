"""Prevent reintroducing the real T4 crash: PeftModel.set_adapter(list) is invalid."""
import pytest
from h167.combined_probe import activate_frozen_language_and_trainable_vision

class FakeLoraTuner:
    def __init__(self):
        self.active_adapters=["language"]
        self.calls=[]
    def set_adapter(self,names):
        assert names==["language","vision"]
        self.calls.append(names)
        self.active_adapters=list(names)

class FakePeftModel:
    def __init__(self):
        self.peft_config={"language":object(),"vision":object()}
        self.base_model=FakeLoraTuner()
        self.wrapper_calls=0
    @property
    def active_adapters(self):
        return self.base_model.active_adapters
    def set_adapter(self,names):
        self.wrapper_calls+=1
        raise TypeError("PeftModel.set_adapter requires a single string")

def test_multi_adapter_uses_underlying_lora_tuner_never_peft_wrapper():
    model=FakePeftModel()
    assert activate_frozen_language_and_trainable_vision(model)==["language","vision"]
    assert model.wrapper_calls==0
    assert model.base_model.calls==[["language","vision"]]

def test_missing_adapter_fails_before_mutation():
    model=FakePeftModel()
    model.peft_config.pop("vision")
    with pytest.raises(ValueError):
        activate_frozen_language_and_trainable_vision(model)
    assert model.base_model.calls==[]

def test_tuner_claiming_only_one_adapter_is_not_accepted():
    model=FakePeftModel()
    def broken(names):
        model.base_model.active_adapters=["vision"]
    model.base_model.set_adapter=broken
    with pytest.raises(ValueError):
        activate_frozen_language_and_trainable_vision(model)

def test_peft_wrapper_and_tuner_active_sets_must_agree():
    class BrokenWrapper(FakePeftModel):
        @property
        def active_adapters(self):
            return ["language"]
    with pytest.raises(ValueError):
        activate_frozen_language_and_trainable_vision(BrokenWrapper())
