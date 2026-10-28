"""H173 frozen H164+vision OOD must never retain trainable tensors."""
import pytest
from h173.infer_gpu import enforce_entire_model_frozen


class StubParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class StubModel:
    def __init__(self, n):
        self._parameters = [StubParameter() for _ in range(n)]
        self.training = True

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return self._parameters


def test_full_model_frozen_even_if_adapter_was_trainable():
    mock = StubModel(7)
    assert all(p.requires_grad for p in mock.parameters())
    assert enforce_entire_model_frozen(mock) == 7
    assert mock.training is False
    assert all(p.requires_grad is False for p in mock.parameters())


def test_adapter_switch_is_always_refrozen():
    mock = StubModel(4)
    assert enforce_entire_model_frozen(mock) == 4
    # Real PEFT tuner activation may toggle LoRA requires_grad.
    mock.parameters()[2].requires_grad_(True)
    mock.training = True
    assert enforce_entire_model_frozen(mock) == 4
    assert all(not p.requires_grad for p in mock.parameters())
    assert mock.training is False


def test_empty_or_absent_model_parameters_never_qualify():
    with pytest.raises(ValueError, match="trainable weights"):
        enforce_entire_model_frozen(StubModel(0))
