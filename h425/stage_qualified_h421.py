"""Stage only a GENUINELY source-qualified H425 trained LoRA for private H421 evaluation.

This is not a model trainer, source-rights approval, external OOD, or public release.
Never uploads data; Kaggle dataset creation is separate and owner-private.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "abdulazizshehri/hala-h425-genuine-trained-parent-private"
STATUS = "COMPLETE_PRIVATE_H425_H417_REPLAY_TRUE_NATIVE_PAIRED"
H421_NAMES = {
    "receipt.json": "h425_original_model_receipt.json",
    "receipt.sha256": "h425_original_model_receipt.sha256",
    "adapter_model.safetensors": "h425_original_trained_adapter.safetensors",
    "adapter_config.json": "h425_original_trained_adapter_config.json",
}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_verifier():
    path = ROOT / "h425" / "validate_receipt.py"
    spec = importlib.util.spec_from_file_location("h425_staged_source_verifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def demand_true_training(proof, receipt, provider_bytes_sha256):
    """Gate staging on BOTH byte-verified model AND original JPEG source evidence."""
    if not isinstance(proof, dict) or not isinstance(receipt, dict):
        raise ValueError("H425 proof must be actual model/source verifier data")
    for flag in (
        "original_source_only_gate", "stronger_source_only_gate",
        "trained_adapter_sha_verified", "all_eight_intermediate_checkpoint_shas_verified",
        "original_source_media_replayed",
    ):
        if proof.get(flag) is not True:
            raise ValueError("H425 private trained model has no verified " + flag)
    if (
        proof.get("status") != STATUS
        or receipt.get("status") != STATUS
        or receipt.get("schema") != "hala.h425.original-h417-replay-two-prompt-lora.v1"
        or proof.get("actual_optimizer_steps") != 12696
        or proof.get("actual_original_image_conditioned_gradient_passes") != 50784
        or receipt.get("model_weights_sha256") is None
        or proof.get("receipt_sha256") is None
    ):
        raise ValueError("H425 source-original GPU receipt/steps not qualified")
    for key in ("independent_external_ood", "source_rights_review_qualified",
                "release_qualified", "weights_public"):
        if receipt.get(key) is not False:
            raise ValueError("H425 staging forbids any unsupported rights/OOD/release claim")
    if proof["receipt_sha256"] != provider_bytes_sha256:
        raise ValueError("H425 original provider receipt raw-byte hash/proof inconsistent")
    return True


def _assert_config(config):
    if (
        not isinstance(config, dict)
        or config.get("base_model_name_or_path") != "Qwen/Qwen3-VL-2B-Instruct"
        or config.get("peft_type") != "LORA"
        or type(config.get("r")) is not int or config["r"] != 8
        or type(config.get("lora_alpha")) is not int or config["lora_alpha"] != 16
        or set(config.get("target_modules") or []) != {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise ValueError("H425 trained adapter config not original Qwen3-VL LoRA")


def stage_private_h421(receipt_root, source_root, paired_root, h421_root, output_root):
    receipt_root, output_root = Path(receipt_root), Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("H425 private owner dataset stage is strictly create-only")
    if output_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("H425 private model bytes must NEVER be staged under the GitHub repository")
    adapter_dir = receipt_root / "private-h417-replay-paired-native-adapter"
    adapter = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    checkpoints = receipt_root / "private-checkpoints"
    # This verifies 240 ORIGINAL source JPEGs, actual 8 LoRA checkpoint SHA bytes,
    # final trained LoRA SHA and full optimizer/loss/strong-gate contract.
    verify = _load_verifier().verify
    proof = verify(
        receipt_root, adapter_file=adapter, source_root=source_root,
        paired_root=paired_root, h421_root=h421_root,
        checkpoints_root=checkpoints,
    )
    raw_receipt = (receipt_root / "receipt.json").read_bytes()
    original_receipt = json.loads(raw_receipt)
    demand_true_training(proof, original_receipt, hashlib.sha256(raw_receipt).hexdigest())
    _assert_config(json.loads(config_path.read_text("utf8")))
    original = {
        receipt_root / "receipt.json": H421_NAMES["receipt.json"],
        receipt_root / "receipt.sha256": H421_NAMES["receipt.sha256"],
        adapter: H421_NAMES["adapter_model.safetensors"],
        config_path: H421_NAMES["adapter_config.json"],
    }
    for path in original:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing real provider source model byte file " + path.name)
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        for src, name in original.items():
            dst = output_root / name
            shutil.copyfile(src, dst)
            if _sha256(src) != _sha256(dst):
                raise ValueError("H425 copied original model/receipt byte SHA mismatch")
        metadata = {
            "title": "Hala H425 Authenticated Source-Qualified Parent Private",
            "id": DATASET_ID,
            "description": (
                "OWNER PRIVATE H425 source-original H417 parent continuation for H421 "
                "384-action contrastive comparison ONLY. Original provider receipt SHA "
                + proof["receipt_sha256"] + "; genuine trained LoRA adapter SHA "
                + original_receipt["model_weights_sha256"]
                + ". Original 240 source-heldout source-only gate and all 8 checkpoint "
                "hashes verified on owner Windows; NO independent OOD, clinical/GUI "
                "automation approval, rights review or public model release."
            ),
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }
        (output_root / "dataset-metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2), encoding="utf8"
        )
        if len(list(output_root.iterdir())) != 5:
            raise ValueError("H425 private owner stage must hold exactly 4 source files plus metadata")
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return {
        "status": "STAGED_PRIVATE_ONLY_NOT_UPLOADED",
        "dataset_id": DATASET_ID,
        "receipt_sha256": proof["receipt_sha256"],
        "trained_adapter_sha256": original_receipt["model_weights_sha256"],
        "source_only_qualified": True,
        "independent_external_ood": False,
        "weights_public": False,
        "files": 5,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--receipt-root", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--paired", type=Path, required=True)
    p.add_argument("--h421", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(stage_private_h421(
        args.receipt_root, args.sources, args.paired, args.h421, args.output_root
    ), sort_keys=True))
