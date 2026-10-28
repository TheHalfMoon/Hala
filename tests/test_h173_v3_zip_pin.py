"""H173 v3 immutable original-application ZIP source pin, no owner data."""
from h173.infer_gpu import OOD_ZIP_SHA as INFERENCE_PIN
from h173.audit_gpu import OOD_ZIP_SHA as GPU_AUDIT_PIN
from h173.native_reward import OOD_ZIP_SHA as REWARD_PIN

V3_SHA = "9775b2bfaad88a29e3ce4e1722ba86f417ab579c1d1ab1b8de9721e235266c18"


def test_private_v3_gold_free_ood_pin_is_exact_frozen_sha():
    assert len(V3_SHA) == 64
    assert INFERENCE_PIN == V3_SHA
    assert GPU_AUDIT_PIN == V3_SHA
    assert REWARD_PIN == V3_SHA
