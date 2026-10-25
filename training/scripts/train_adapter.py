"""Unexecuted remote-only Qwen3-VL LoRA candidate; no jobs or billing are created.

Pure artifact/mask/checkpoint helpers have CPU tests. Actual processor/training,
model fit and resume equivalence still require a pinned real remote experiment.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

from training.artifacts import (MAX_PIXELS, artifact_path, load_sft_artifact,
                                sha256_file, strict_json)
from h48.factory import SCHEMA as H48_SOURCE_SCHEMA
from h48.sft_bridge import load_export as load_h48_export
from h49.factory import SCHEMA as H49_SOURCE_SCHEMA
from h49.audit import load_export as load_h49_export
from h50.counterfactual import SCHEMA as H50_SOURCE_SCHEMA, load_export as load_h50_export
from h52.factory import VERSION as H52_SOURCE_SCHEMA
from h52.training_bridge import load_export as load_h52_export
from h54.native_drag import (SCHEMA as H54_DRAG_SCHEMA,
                             load_export as load_h54_drag_export,
                             validate_drag_baseline)
from h57.native_multidomain_sft import (SCHEMA as H57_DRAG_SCHEMA,
                                       load_export as load_h57_drag_export,
                                       validate_mixed_drag_baseline)
from h60.native_drag_sft import (SCHEMA as H60_DRAG_SCHEMA,
                                 load_export as load_h60_drag_export,
                                 validate_drag_baseline as validate_h60_drag_baseline)
from h61.curriculum import SCHEMA as H61_DRAG_SCHEMA, load_export as load_h61_drag_export
from h61.supervision import completion_weights, weighted_lm_loss
from h75.processor_audit import select_processor_audit_rows, audit_summary
from training.checkpoints import (IDENTITY_FIELDS, atomic_json, completed_checkpoint_result,
                                  load_checkpoint, publish_checkpoint)
from training.masking import completion_labels
from training.preflight import TrainingConfig, qualify_environment, validate_experiment
from h85.model_cache import HF_HOME, pinned_cache_kwargs
from h99.sampler import STRATEGY as H99_SAMPLER, build_balanced_plan
from h106.training_plan import STRATEGY as H106_SAMPLER, get_plan as get_h106_joint_plan
from training.scripts.gpu_probe import probe
from training.scripts.baseline import reject_kaggle_runtime

LORA_TARGET_PATTERN = r'.*language_model.*\.(q_proj|v_proj)$'
# H96 isolated research only. Vision last-four blocks: 8 explicit fused
# QKV/output projections; the original text-only adapter remains unchanged.
VISION_LORA_TARGET_PATTERN = r'.*visual\.blocks\.(20|21|22|23)\.attn\.(qkv|proj)$'
VISION_LORA_SCOPE_NONCE = 'hala.h96.vision-last-four-lora-rank8-v1'
# Opt-in isolated original H105 TRAIN pilot; H99 and language-only paths unchanged.
JOINT_LORA_TARGET_PATTERN = (r'.*(?:language_model.*\.(?:q_proj|v_proj)|'
                             r'visual\.blocks\.(?:20|21|22|23)\.attn\.(?:qkv|proj))$')
JOINT_LORA_SCOPE_NONCE = 'hala.h106.original-h105-joint-vision-language-lora-rank8-v1'
H114_POINT_SCOPE_NONCE = 'hala.h114.original-h113-independent-start-end-point-fresh-joint-lora-v1'
H117_DELTA_SCOPE_NONCE = 'hala.h117.original-h116-conditioned-signed-end-delta-fresh-joint-lora-v1'
H119_FULL512_SCOPE_NONCE = 'hala.h119.full512-anchor-robust-1024-fresh-joint-lora-v1'
H122_GRID_SCOPE_NONCE = 'hala.h122.original-train448-end-grid10-finepoint-joint-lora-v1'


def qualify_vision_target_names(names):
    expected = {f'visual.blocks.{i}.attn.{part}'
                for i in range(20, 24) for part in ('qkv', 'proj')}
    matched = [name for name in names
               if re.fullmatch(VISION_LORA_TARGET_PATTERN, name)]
    suffixes = [name[name.index('visual.blocks.'):] for name in matched]
    if len(matched) != 8 or set(suffixes) != expected or len(set(suffixes)) != 8:
        raise ValueError('Exactly eight Qwen3-VL late visual QKV/proj layers required')
    return tuple(sorted(matched))


def qualify_joint_target_names(names):
    """Exact 8 late visual layers plus real language-model q/v projections."""
    names=tuple(names)
    visual=qualify_vision_target_names(names)
    language=tuple(sorted(name for name in names
                          if re.fullmatch(LORA_TARGET_PATTERN,name)))
    if not 16<=len(language)<=256 or len(set(language))!=len(language):
        raise ValueError('H106 needs bounded real Qwen3-VL language-model q/v layers')
    targeted=tuple(sorted((*visual,*language)))
    if any(re.fullmatch(JOINT_LORA_TARGET_PATTERN,name) is None for name in targeted):
        raise ValueError('H106 joint PEFT regex would miss a validated target')
    if len(set(targeted))!=len(targeted):
        raise ValueError('H106 duplicate visual/language target rejected')
    return targeted,visual,language


class QualificationInterrupted(RuntimeError):
    """Deliberate stop after the atomic paired snapshot, before finalization."""
    def __init__(self, step, version):
        super().__init__(f'Qualification stopped after published step {step}')
        self.step, self.version = step, Path(version)


def interrupt_after_checkpoint(stop_after_step, step, version):
    if stop_after_step == step:
        raise QualificationInterrupted(step, version)


def create_training_scaler(torch, precision):
    # The bounded FP16 smoke must validate its first gradients, not skip an
    # overflowing update while the default 65536 scale backs off. Resume below
    # restores the saved scaler state instead of restarting this initial scale.
    return torch.amp.GradScaler('cuda', init_scale=1.0, enabled=precision == 'fp16')


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Bounded experiment JSON required')
    return strict_json(path.read_text(encoding='utf-8'))


def media_format(row, native_drag_jpeg=False) -> str:
    """The image format of one training record.

    A record that declares its own format is trusted; a record that does not
    falls back to the source-wide flag, which is what the earlier sources
    need. A mixed corpus therefore does not have to lie about its format.
    """
    declared = row.get('image_format')
    if declared in ('JPEG', 'PNG'):
        return declared
    return 'JPEG' if native_drag_jpeg else 'PNG'


def check_media(row, root, native_drag_jpeg=False):
    """Verify one training media file without allocating a processor.

    The media digest and the format are checked for EVERY record, and the
    processor is only needed for the records that are actually audited. That
    split is what makes a curriculum of a hundred thousand records affordable
    without weakening the integrity of the corpus.
    """
    from PIL import Image

    image_path = artifact_path(root, row['image_path'])
    if sha256_file(image_path) != row['image_sha256']:
        raise ValueError('Training media digest changed after preflight')
    with Image.open(image_path) as source:
        if source.format != media_format(row, native_drag_jpeg):
            raise ValueError('Training image does not have its declared format')
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('Training image exceeds source-bound header/pixel budget')


def prepare_inputs(row, processor, root, token_budget, *, native_drag_jpeg=False,
                   supervision='completion', coordinate_weight=8.0, return_weights=False,
                   h117_target_only=False, h122_target_only=False):
    """One bounded image; validate expanded vision/prompt prefix before GPU use.

    The return shape stays `(inputs, labels)` unless `return_weights` is
    requested, so every existing caller is unaffected.
    """
    import torch
    from PIL import Image
    image_path = artifact_path(root, row['image_path'])
    if sha256_file(image_path) != row['image_sha256']:
        raise ValueError('Training media digest changed after preflight')
    with Image.open(image_path) as source:
        if source.format != media_format(row, native_drag_jpeg):
            raise ValueError('Training image does not have its declared format')
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('Training image exceeds source-bound header/pixel budget')
        image = source.convert('RGB')
    try:
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(
            messages + [{'role': 'assistant', 'content': row['completion']}],
            tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[full], images=[image], return_tensors='pt')
        prompt_inputs = processor(text=[prefix], images=[image], return_tensors='pt')
        if inputs.input_ids.shape[0] != 1 or prompt_inputs.input_ids.shape[0] != 1:
            raise ValueError('Only candidate batch-size-one masking is implemented')
        label_ids = completion_labels(inputs.input_ids[0].tolist(),
                                      prompt_inputs.input_ids[0].tolist(), token_budget)
        labels = torch.tensor([label_ids], dtype=inputs.input_ids.dtype)
        if supervision == 'coordinate_weighted':
            completion_ids = label_ids[len(prompt_inputs.input_ids[0].tolist()):]
            if h122_target_only:
                from h122.supervision import coarse_fine_weights
                if row.get('endpoint_role')!='end':
                    raise ValueError('H122 exclusively original END spatial point')
                weights = coarse_fine_weights(
                    completion_ids,processor.tokenizer.decode,coordinate_weight)
            elif h117_target_only:
                from h117.supervision import target_only_weights
                weights = target_only_weights(
                    completion_ids,processor.tokenizer.decode,
                    row['endpoint_role'],coordinate_weight)
            else:
                weights = completion_weights(completion_ids, processor.tokenizer.decode,
                                             coordinate_weight)
            weight_values = [0.0] * len(prompt_inputs.input_ids[0].tolist()) + weights
        else:
            weight_values = [0.0 if token == -100 else 1.0 for token in label_ids]
        if not return_weights:
            return inputs, labels
        weights = torch.tensor([weight_values], dtype=torch.float32)
        return inputs, labels, weights
    finally:
        image.close()


def load_authoring_records(examples_path, manifest):
    """Verify original H48/H49/H50/H52/H54/H57/H60/H61 sources; H14 remains frozen."""
    if manifest.get('source_schema') == H61_DRAG_SCHEMA:
        return load_h61_drag_export(examples_path, manifest)
    if manifest.get('source_schema') == H60_DRAG_SCHEMA:
        return load_h60_drag_export(examples_path, manifest)
    if manifest.get('source_schema') == H57_DRAG_SCHEMA:
        return load_h57_drag_export(examples_path, manifest)
    if manifest.get('source_schema') == H54_DRAG_SCHEMA:
        return load_h54_drag_export(examples_path, manifest)
    if manifest.get('source_schema') == H52_SOURCE_SCHEMA:
        return load_h52_export(examples_path, manifest)
    if manifest.get('source_schema') == H50_SOURCE_SCHEMA:
        return load_h50_export(examples_path, manifest)
    if manifest.get('source_schema') == H49_SOURCE_SCHEMA:
        return load_h49_export(examples_path, manifest)
    if manifest.get('source_schema') == H48_SOURCE_SCHEMA:
        return load_h48_export(examples_path, manifest)
    return load_sft_artifact(examples_path, manifest)


def train(config_path, examples_path, manifest_path, baseline_path, output_path, resume=False,
          stop_after_step=None, init_adapter=None, vision_lora=False,
          sampling_strategy=None, joint_lora=False, point_sft_path=None,
          conditioned_delta_sft_path=None, robust_anchor_sft_path=None,
          end_grid_sft_path=None):
    # H96 is a distinct isolated pilot. Do not silently reinterpret earlier
    # language-only checkpoints or combine adapters without qualification.
    if type(vision_lora) is not bool or type(joint_lora) is not bool:
        raise ValueError('Explicit visual and joint LoRA boolean selectors required')
    if vision_lora and joint_lora:
        raise ValueError('H96 visual-only and H106 joint LoRA paths cannot mix')
    if point_sft_path is not None and (
            not joint_lora or sampling_strategy!=H106_SAMPLER
            or resume or init_adapter is not None):
        raise ValueError('H114 isolated fresh joint point SFT must use immutable H106 source selection')
    if conditioned_delta_sft_path is not None and (
            point_sft_path is not None or not joint_lora
            or sampling_strategy!=H106_SAMPLER
            or resume or init_adapter is not None):
        raise ValueError('H117 separate original TRAIN-only conditioned signed delta LoRA required')
    if end_grid_sft_path is not None and (
            type(end_grid_sft_path) not in (str,Path)
            or point_sft_path is not None or conditioned_delta_sft_path is not None
            or robust_anchor_sft_path is not None or not joint_lora
            or sampling_strategy!=H106_SAMPLER or resume
            or init_adapter is not None or vision_lora):
        raise ValueError('H122 fresh original 448/64 END-only grid SFT isolated path')
    if end_grid_sft_path is not None and stop_after_step is not None:
        from h122.training_plan import checkpoint_due as due122
        if not due122(stop_after_step):
            raise ValueError('H122 only 64-step durable checkpoint interruption')
    if robust_anchor_sft_path is not None and (
            type(robust_anchor_sft_path) not in (str, Path)
            or point_sft_path is not None or conditioned_delta_sft_path is not None
            or not joint_lora or sampling_strategy!=H106_SAMPLER
            or resume or init_adapter is not None or vision_lora):
        raise ValueError('H119 isolated fresh TRAIN-only full512/1024 path required')
    if robust_anchor_sft_path is not None and stop_after_step is not None:
        from h119.training_plan import checkpoint_due
        if not checkpoint_due(stop_after_step):
            raise ValueError('H119 interruption only allowed at durable 64-step checkpoint')
    if sampling_strategy not in (None, H99_SAMPLER, H106_SAMPLER):
        raise ValueError('only pinned original H99 or H106 source plans are admitted')
    if sampling_strategy==H99_SAMPLER and (not vision_lora or init_adapter is not None or resume):
        raise ValueError('H99 remains fresh vision-only research')
    if sampling_strategy==H106_SAMPLER and (not joint_lora or init_adapter is not None or resume):
        raise ValueError('H106 is isolated fresh joint visual-language research')
    if joint_lora and sampling_strategy!=H106_SAMPLER:
        raise ValueError('H106 joint LoRA requires its original H105 train prefix')
    if vision_lora and (init_adapter is not None or resume):
        raise ValueError('H96 first pilot forbids warmstart/resume or combined LoRA')
    config_data = load_json(config_path)
    config = TrainingConfig.from_dict(config_data)
    from h87.research_runtime import require_training_provider
    research_provider = require_training_provider(config)
    if stop_after_step is not None and (type(stop_after_step) is not int or
                                       not 1 <= stop_after_step < config.max_steps):
        raise ValueError('Interruption must leave a remaining optimizer step')
    if config.batch_size != 1:
        raise ValueError('Only candidate batch-size-one accumulation is implemented')
    if vision_lora and (config.platform != 'kaggle' or
                        config.max_steps > (128 if sampling_strategy else 32)
                        or config.lora_rank > 8 or config.precision != 'fp16'
                        or config.gradient_accumulation != 1):
        raise ValueError('H96 vision study bounded to <=32 steps, rank<=8, one T4 microbatch')
    if joint_lora and (config.platform!='kaggle' or config.max_steps!=(448 if end_grid_sft_path is not None else (1024 if robust_anchor_sft_path is not None else 256))
                       or config.lora_rank!=8 or config.precision!='fp16'
                       or config.gradient_accumulation!=1 or config.batch_size!=1
                       or config.supervision!='coordinate_weighted'
                       or config.coordinate_weight!=8.0):
        raise ValueError('H106 only exact noncommercial 256-step fp16 rank8 weighted joint pilot')
    if research_provider != 'kaggle_noncommercial_research':
        reject_kaggle_runtime()
    if config.model_id != 'Qwen/Qwen3-VL-2B-Instruct':
        raise ValueError('Only the candidate Qwen3-VL processor path is implemented')
    manifest = load_json(manifest_path)
    baseline = load_json(baseline_path)
    if manifest.get('source_schema') == H54_DRAG_SCHEMA:
        validate_drag_baseline(baseline)
    if manifest.get('source_schema') == H57_DRAG_SCHEMA:
        validate_mixed_drag_baseline(baseline)
    if manifest.get('source_schema') == H60_DRAG_SCHEMA:
        validate_h60_drag_baseline(baseline)
    if manifest.get('source_schema') == H61_DRAG_SCHEMA:
        validate_h60_drag_baseline(baseline)
    validate_experiment(config, manifest, baseline)
    records = load_authoring_records(examples_path, manifest)
    root = Path(examples_path).parent.resolve()
    # Do not reorder, overwrite or republish the original 10k immutable H71
    # curriculum. Selection is an opt-in index plan bound to training identity.
    if sampling_strategy==H99_SAMPLER:
        sampling_plan=build_balanced_plan(
            records, config.max_steps * config.gradient_accumulation)
        if set(sampling_plan.summary['selected_languages']) != {'ar','en'}:
            raise ValueError('H99 sampler lost original bilingual coverage')
    elif sampling_strategy==H106_SAMPLER:
        if manifest.get('source_schema')!=H61_DRAG_SCHEMA:
            raise ValueError('H106 only immutable original H71 TRAIN source')
        if end_grid_sft_path is not None:
            from h122.schedule import qualify_owner_private as qualify_h122_source
            from h122.training_plan import get_plan as get_h122_plan
            h122_grid_plan=qualify_h122_source(Path(end_grid_sft_path))
            sampling_plan=get_h122_plan(records,root,h122_grid_plan,config.max_steps)
        elif robust_anchor_sft_path is not None:
            from h119.training_plan import get_plan as get_h119_full_plan
            sampling_plan=get_h119_full_plan(records,root,config.max_steps)
        else:
            sampling_plan=get_h106_joint_plan(records,root,config.max_steps)
        if sampling_plan.summary['source_split']!='TRAIN_ONLY':
            raise ValueError('H106 no EVAL or TEST in training')
    else:
        sampling_plan=None
    h114_point_plan=None
    if end_grid_sft_path is None:
        h122_grid_plan=None
    if point_sft_path is not None:
        from h114.schedule import qualify_owner_private,SCREENSHOTS,STEPS
        if (manifest.get('source_schema')!=H61_DRAG_SCHEMA
                or config.max_steps!=STEPS
                or sampling_plan is None or len(sampling_plan.indices)!=STEPS):
            raise ValueError('H114 requires exact H71 original TRAIN and 256-step H105 sampler')
        h114_point_plan=qualify_owner_private(Path(point_sft_path))
        for image_index in range(SCREENSHOTS):
            original=records[sampling_plan.indices[image_index]]
            for role_index in range(2):
                derived=h114_point_plan.selected[image_index*2+role_index]
                if (derived['original_source_id']!=original['id']
                        or derived['image_sha256']!=original['image_sha256']
                        or derived['image_path']!=original['image_path']
                        or derived['language']!=original['language']
                        or derived['family']!=original['family']):
                    raise ValueError('H114 original point source does not equal original H105 TRAIN source')
    h117_delta_plan=None
    if conditioned_delta_sft_path is not None:
        from h114.schedule import qualify_owner_private as qualify_h113_points, SCREENSHOTS, STEPS
        from h116.delta import derive_schedule
        if (manifest.get('source_schema')!=H61_DRAG_SCHEMA
                or config.max_steps!=STEPS or sampling_plan is None
                or len(sampling_plan.indices)!=STEPS):
            raise ValueError('H117 requires original H71 TRAIN-only exact 256-step H105 source prefix')
        parent=qualify_h113_points(Path(conditioned_delta_sft_path))
        h117_delta_plan=derive_schedule(parent)
        if h117_delta_plan.summary['derived_private_training_jsonl_sha256']!=(
                '174e16634b317ee23a4b6661568f51721a0ca72b6cdabd83d9f107bf361b3d7c'):
            raise ValueError('H116 original TRAIN derived signed displacement source digest mismatch')
        for image_index in range(SCREENSHOTS):
            original=records[sampling_plan.indices[image_index]]
            for role_index in range(2):
                derived=h117_delta_plan.selected[image_index*2+role_index]
                if (derived['original_source_id']!=original['id']
                        or derived['image_sha256']!=original['image_sha256']
                        or derived['image_path']!=original['image_path']
                        or derived['language']!=original['language']
                        or derived['family']!=original['family']):
                    raise ValueError('H117 original H116 point and signed displacement source mismatched H71')
    h119_full_plan=None
    if robust_anchor_sft_path is not None:
        from h119.curriculum import qualify_owner_private as qualify_h119_source
        from h119.training_plan import STEPS as H119_STEPS, PAIRS as H119_PAIRS
        if (manifest.get('source_schema')!=H61_DRAG_SCHEMA or
                config.max_steps!=H119_STEPS or sampling_plan is None or
                len(sampling_plan.indices)!=H119_PAIRS):
            raise ValueError('H119 requires original 512 H105 selected TRAIN screenshots')
        h119_full_plan=qualify_h119_source(Path(robust_anchor_sft_path))
        if h119_full_plan.summary['source_split']!='TRAIN_ONLY':
            raise ValueError('H119 source not TRAIN-only')
        for image_index in range(H119_PAIRS):
            original=records[sampling_plan.indices[image_index]]
            for role_index in range(2):
                derived=h119_full_plan.selected[image_index*2+role_index]
                if (derived['original_source_id']!=original['id'] or
                        derived['image_sha256']!=original['image_sha256'] or
                        derived['image_path']!=original['image_path'] or
                        derived['language']!=original['language'] or
                        derived['family']!=original['family'] or
                        derived['split']!='train'):
                    raise ValueError('H119 source image/prompt does not match exact original H105 TRAIN')
    if h122_grid_plan is not None:
        from h122.grid import verify_overlay
        from h122.schedule import TRAIN as H122_TRAIN,HOLDOUT as H122_HOLDOUT
        if (config.max_steps!=H122_TRAIN or
                len(sampling_plan.indices)!=H122_TRAIN or
                h122_grid_plan.summary.get('heldout_train_source_images')!=H122_HOLDOUT):
            raise ValueError('H122 exact 448 TRAIN /64 heldout no optimizer overlap')
        for ordinal,(index,derived) in enumerate(zip(
                sampling_plan.indices,h122_grid_plan.train)):
            if not verify_overlay(records[index],derived):
                raise ValueError('H122 original source END target changed')
    native_drag = manifest.get('source_schema') in (H54_DRAG_SCHEMA, H57_DRAG_SCHEMA,
                                                   H60_DRAG_SCHEMA, H61_DRAG_SCHEMA)
    signature = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
    if vision_lora:
        signature = hashlib.sha256((signature + VISION_LORA_SCOPE_NONCE).encode()).hexdigest()
    if joint_lora:
        signature=hashlib.sha256((signature+JOINT_LORA_SCOPE_NONCE).encode()).hexdigest()
    if sampling_plan is not None:
        signature = hashlib.sha256(
            (signature + sampling_strategy + sampling_plan.sha256).encode()).hexdigest()
    if h114_point_plan is not None:
        signature = hashlib.sha256((
            signature + H114_POINT_SCOPE_NONCE +
            h114_point_plan.summary['original_h113_point_jsonl_sha256'] +
            h114_point_plan.summary['selected_identities_anonymized_sha256']).encode()).hexdigest()
    if h117_delta_plan is not None:
        signature=hashlib.sha256((
            signature + H117_DELTA_SCOPE_NONCE +
            h117_delta_plan.summary['original_h113_parent_sha256'] +
            h117_delta_plan.summary['derived_private_training_jsonl_sha256']
        ).encode()).hexdigest()
    if h122_grid_plan is not None:
        signature=hashlib.sha256((
            signature + H122_GRID_SCOPE_NONCE +
            h122_grid_plan.summary['train_private_sha256'] +
            h122_grid_plan.summary['heldout_private_sha256'] +
            sampling_plan.sha256).encode()).hexdigest()
    if h119_full_plan is not None:
        signature=hashlib.sha256((
            signature + H119_FULL512_SCOPE_NONCE +
            h119_full_plan.summary['derived_private_train_sha256'] +
            sampling_plan.sha256).encode()).hexdigest()
    init_digest = ''
    if init_adapter:
        init_path = Path(init_adapter)
        adapter_file = init_path / 'adapter_model.safetensors'
        if not init_path.is_dir() or not adapter_file.exists():
            raise ValueError('An initialised adapter directory is required')
        init_digest = sha256_file(adapter_file)
        # The training identity binds the initial adapter, so a resumed run can
        # never continue from a different starting adapter than the one the
        # checkpoint was published against.
        signature = hashlib.sha256((signature + init_digest).encode()).hexdigest()
    identity = {'config_sha256': signature, 'dataset_sha256': manifest['dataset_sha256'],
                'manifest_sha256': sha256_file(manifest_path), 'max_steps': config.max_steps}
    output = Path(output_path)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Refusing to overwrite existing experiment output')
    output.mkdir(parents=True, exist_ok=True)
    previous, version = None, None
    if resume:
        # H82: a killed free-T4 runtime can leave a fully durable optimizer
        # version after atomic rename, but before its latest.json publish.
        # Recover exactly one verified next-step version, not a new fork.
        from h82.orphan_recovery import recover_orphaned_checkpoint
        recover_orphaned_checkpoint(output, identity)
        version, previous = load_checkpoint(output, identity)
        if previous['step'] == config.max_steps:
            # Finalization after interruption needs no further weights or optimizer step.
            result = completed_checkpoint_result(version, previous, config_data)
            atomic_json(output / 'training-result.json', result)
            return result
    runtime = probe()
    qualify_environment(config, runtime)
    for package, expected in (('torch', '2.11.0'), ('transformers', '5.16.1'), ('peft', '0.20.0')):
        actual = runtime['packages'][package]
        if actual is None or actual.split('+')[0] != expected:
            raise ValueError(f'Unqualified dependency version: {package}')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['HF_HOME'] = HF_HOME
    import torch
    import peft
    import transformers
    import PIL
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    runtime['imported_packages'] = {'torch': torch.__version__,
                                    'transformers': transformers.__version__,
                                    'peft': peft.__version__, 'pillow': PIL.__version__}
    for package, expected in (('torch', '2.11.0'), ('transformers', '5.16.1'),
                              ('peft', '0.20.0')):
        if runtime['imported_packages'][package].split('+')[0] != expected:
            raise ValueError(f'Imported dependency version mismatch: {package}')

    torch.manual_seed(42)
    dtype = torch.float16 if config.precision == 'fp16' else torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        config.model_id, **pinned_cache_kwargs(config.model_id, config.revision))
    # Every record is checked for its media digest and its format. The
    # processor audit is bounded so that a curriculum of a hundred thousand
    # records does not cost hours before the first optimizer step.
    for row in records:
        check_media(row, root, native_drag_jpeg=native_drag)
    audit_rows = select_processor_audit_rows(records, config.audit_records)
    for row in audit_rows:
        checked_inputs, checked_labels, checked_weights = prepare_inputs(
            row, processor, root, manifest['token_budget'], native_drag_jpeg=native_drag,
            supervision=config.supervision, coordinate_weight=config.coordinate_weight,
            return_weights=True)
        del checked_inputs, checked_labels, checked_weights
    runtime['audited_records'] = len(audit_rows)
    if sampling_strategy==H99_SAMPLER:
        runtime['h99_training_sampler'] = sampling_plan.summary
    if sampling_strategy==H106_SAMPLER:
        runtime['h106_joint_training_sampler'] = sampling_plan.summary
    if h114_point_plan is not None:
        from h114.schedule import trainer_row
        runtime['h114_independent_point_sampler'] = h114_point_plan.summary
        for ordinal in (0,1,2,3,126,127,252,253,254,255):
            base=records[sampling_plan.indices[ordinal//2]]
            derived=h114_point_plan.selected[ordinal]
            audited=trainer_row(base,derived)
            prepare_inputs(audited,processor,root,manifest['token_budget'],
                           native_drag_jpeg=native_drag,
                           supervision=config.supervision,
                           coordinate_weight=config.coordinate_weight,
                           return_weights=True)
        runtime['h114_point_processor_audited_roles'] = 10
    if h117_delta_plan is not None:
        from h114.schedule import trainer_row
        runtime['h117_start_conditioned_delta_sampler'] = h117_delta_plan.summary
        for ordinal in (0,1,2,3,126,127,252,253,254,255):
            base=records[sampling_plan.indices[ordinal//2]]
            derived=h117_delta_plan.selected[ordinal]
            checked=trainer_row(base,derived)
            prepare_inputs(checked,processor,root,manifest['token_budget'],
                           native_drag_jpeg=native_drag,
                           supervision=config.supervision,
                           coordinate_weight=config.coordinate_weight,
                           return_weights=True,h117_target_only=True)
        runtime['h117_signed_delta_processor_audited_roles'] = 10
    if h122_grid_plan is not None:
        from h114.schedule import trainer_row
        runtime['h122_end_grid_train_sampler']=h122_grid_plan.summary
        runtime['h122_end_grid_original_train_plan']=sampling_plan.summary
        for ordinal in (0,1,2,223,224,445,446,447):
            source=records[sampling_plan.indices[ordinal]]
            role=h122_grid_plan.train[ordinal]
            checked=trainer_row(source,role)
            checked['endpoint_role']='end'
            prepare_inputs(checked,processor,root,manifest['token_budget'],
                           native_drag_jpeg=native_drag,
                           supervision=config.supervision,
                           coordinate_weight=config.coordinate_weight,
                           return_weights=True,h122_target_only=True)
        runtime['h122_end_processor_audits']=8
    if h119_full_plan is not None:
        from h114.schedule import trainer_row
        runtime['h119_full512_robust_anchor_sampler']=h119_full_plan.summary
        runtime['h119_full512_original_plan']=sampling_plan.summary
        for ordinal in (0,1,2,3,254,255,510,511,1018,1019,1022,1023):
            original=records[sampling_plan.indices[ordinal//2]]
            derived=h119_full_plan.selected[ordinal]
            checked=trainer_row(original,derived)
            checked['endpoint_role']=derived['endpoint_role']
            prepare_inputs(checked,processor,root,manifest['token_budget'],
                           native_drag_jpeg=native_drag,
                           supervision=config.supervision,
                           coordinate_weight=config.coordinate_weight,
                           return_weights=True,h117_target_only=True)
        runtime['h119_original_role_processor_audits']=12
    runtime['media_checked_records'] = len(records)
    runtime.update(audit_summary(audit_rows, len(records)))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id, **pinned_cache_kwargs(config.model_id, config.revision),
        torch_dtype=dtype, attn_implementation='sdpa').to('cuda:0')
    vision_modules = ()
    joint_modules = ()
    if vision_lora:
        vision_modules = qualify_vision_target_names(
            name for name, module in model.named_modules())
        modules = dict(model.named_modules())
        if any(not isinstance(modules[name], torch.nn.Linear)
               for name in vision_modules):
            raise ValueError('Vision target must be a real fused attention Linear layer')
    if joint_lora:
        joint_modules,vision_modules,joint_language_modules=qualify_joint_target_names(
            name for name, module in model.named_modules())
        modules = dict(model.named_modules())
        if any(not isinstance(modules[name], torch.nn.Linear)
               for name in joint_modules):
            raise ValueError('H106 combined vision/language targets must be real Linear')
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_rank * 2,
                      target_modules=(JOINT_LORA_TARGET_PATTERN if joint_lora
                                      else (VISION_LORA_TARGET_PATTERN if vision_lora
                                            else LORA_TARGET_PATTERN)),
                      lora_dropout=0, task_type='CAUSAL_LM', bias='none')
    model = (PeftModel.from_pretrained(model, version / 'adapter', is_trainable=True)
             if resume else
             (PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
              if init_adapter else get_peft_model(model, lora)))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if vision_lora:
        if (len(trainable) != 2 * len(vision_modules)
                or any('lora_' not in name or '.visual.blocks.' not in name
                       for name, _ in trainable)
                or any(not any(target in name for target in vision_modules)
                       for name, _ in trainable)):
            raise ValueError('H96 only eight visual last-four LoRA layers may train')
        runtime['h96_vision_target_modules'] = list(vision_modules)
        runtime['h96_vision_lora_only'] = True
    elif joint_lora:
        if (len(trainable)!=2*len(joint_modules)
                or any('lora_' not in name for name,_ in trainable)
                or any(not any(target in name for target in joint_modules)
                       for name,_ in trainable)):
            raise ValueError('H106 PEFT optimizer modules differ from exact joint targets')
        runtime['h106_joint_lora_verified']=True
        runtime['h106_trainable_visual_modules']=list(vision_modules)
        runtime['h106_trainable_language_modules']=list(joint_language_modules)
        runtime['h106_vision_and_language_unfrozen']=True
    elif not trainable or any('lora_' not in name or 'visual' in name
                              for name, _ in trainable):
        raise ValueError('Only language LoRA parameters may receive gradients; base/vision frozen')
    parameters = [parameter for _, parameter in trainable]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    scaler = create_training_scaler(torch, config.precision)
    start_step = previous['step'] if previous else 0
    metrics = list(previous['metrics']) if previous else []
    changed = previous['changed_parameter_tensors'] if previous else 0
    peak_vram = previous['peak_vram_bytes'] if previous else 0
    if resume:
        state = torch.load(version / 'resume.pt', map_location='cpu', weights_only=True)
        required = {*IDENTITY_FIELDS, 'step', 'optimizer', 'scaler', 'torch_rng', 'cuda_rng'}
        if (not isinstance(state, dict) or set(state) != required or
                any(state[name] != identity[name] for name in IDENTITY_FIELDS) or
                state['step'] != start_step):
            raise ValueError('Resume optimizer and checkpoint identity mismatch')
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    initial_parameters = [parameter.detach().cpu().clone() for parameter in parameters]
    runtime['parameter_residency'] = {}
    for name, parameter in model.named_parameters():
        group = 'adapter' if parameter.requires_grad else 'frozen_base'
        key = f'{group}:{parameter.device}:{parameter.dtype}'
        runtime['parameter_residency'][key] = runtime['parameter_residency'].get(key, 0) + parameter.numel()
    runtime['trainable_parameter_tensors'] = len(parameters)
    runtime['trainable_parameter_elements'] = sum(p.numel() for p in parameters)
    runtime['qualification_steps'] = list(previous['runtime'].get('qualification_steps', [])) if previous else []
    runtime['actual_reload'] = bool(resume)
    runtime['resumed_from_step'] = start_step
    torch.cuda.reset_peak_memory_stats(0)
    for step in range(start_step, config.max_steps):
        step_before = [parameter.detach().cpu().clone() for parameter in parameters]
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        losses = []
        microbatches = []
        for accumulation in range(config.gradient_accumulation):
            ordinal = step * config.gradient_accumulation + accumulation
            point_mode=(h114_point_plan is not None or h117_delta_plan is not None or h119_full_plan is not None or h122_grid_plan is not None)
            index = (sampling_plan.indices[ordinal] if h122_grid_plan is not None
                     else (sampling_plan.indices[ordinal//2] if point_mode
                     else (sampling_plan.indices[ordinal] if sampling_plan is not None
                           else ordinal % len(records))))
            selected_point_rows=(h114_point_plan.selected if h114_point_plan is not None
                                 else (h117_delta_plan.selected if h117_delta_plan is not None
                                       else (h119_full_plan.selected if h119_full_plan is not None else
                                             (h122_grid_plan.train if h122_grid_plan is not None else None))))
            training_row = (trainer_row(records[index],selected_point_rows[ordinal])
                            if point_mode else records[index])
            if h119_full_plan is not None or h122_grid_plan is not None:
                training_row['endpoint_role']=selected_point_rows[ordinal]['endpoint_role']
            inputs, labels, weights = prepare_inputs(training_row, processor, root,
                                                    manifest['token_budget'],
                                                    native_drag_jpeg=native_drag,
                                                    supervision=config.supervision,
                                                    coordinate_weight=config.coordinate_weight,
                                                    return_weights=True,
                                                    h117_target_only=(h117_delta_plan is not None or h119_full_plan is not None),
                                                    h122_target_only=(h122_grid_plan is not None))
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            weights = weights.to('cuda:0')
            microbatches.append({'record_id': training_row['id'],
                                 'input_tokens': int(inputs.input_ids.numel()),
                                 'completion_tokens': int((labels != -100).sum().item()),
                                 'coordinate_tokens': int((weights > 1.0).sum().item()),
                                 'input_devices': {key: str(value.device) for key, value in inputs.items()
                                                   if hasattr(value, 'device')}})
            with torch.autocast('cuda', dtype=dtype):
                if config.supervision == 'coordinate_weighted':
                    outputs = model(**inputs)
                    loss = weighted_lm_loss(outputs.logits, labels, weights)
                else:
                    loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss / config.gradient_accumulation).backward()
            losses.append(loss.item())
        scaler.unscale_(optimizer)
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in parameters):
            raise ValueError('Nonfinite adapter gradient')
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1).item()
        if not __import__('math').isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError('Finite nonzero adapter gradient norm required')
        scaler.step(optimizer)
        scaler.update()
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise ValueError('Nonfinite updated adapter parameter')
        torch.cuda.synchronize(0)
        current_changed = sum(not torch.equal(before, after.detach().cpu())
                              for before, after in zip(initial_parameters, parameters))
        changed = max(changed, current_changed)
        if not changed:
            raise ValueError('No verifiable adapter gradient update')
        step_changed = sum(not torch.equal(before, after.detach().cpu())
                           for before, after in zip(step_before, parameters))
        if not step_changed:
            raise ValueError('No adapter update at this optimizer step')
        free, total = torch.cuda.mem_get_info(0)
        runtime['qualification_steps'].append({
            'step': step + 1, 'gradient_norm_before_clip': gradient_norm,
            'microbatches': microbatches,
            'changed_parameter_tensors_this_step': step_changed,
            'max_parameter_update': max((after.detach().cpu() - before).abs().max().item()
                                        for before, after in zip(step_before, parameters)),
            'allocated_bytes': torch.cuda.memory_allocated(0),
            'reserved_bytes': torch.cuda.memory_reserved(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'device_free_bytes': free, 'device_total_bytes': total})
        metrics.append({'step': step + 1, 'loss': sum(losses) / len(losses),
                        'seconds': time.perf_counter() - start})
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(0))
        progress = {**identity, 'step': step + 1, 'metrics': list(metrics),
                    'changed_parameter_tensors': changed, 'peak_vram_bytes': peak_vram,
                    'runtime': runtime}

        def write_version(stage):
            model.save_pretrained(stage / 'adapter', safe_serialization=True)
            torch.save({**{name: identity[name] for name in IDENTITY_FIELDS},
                        'step': step + 1, 'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all()}, stage / 'resume.pt')

        # H119's 1024 updates must not create 1024 full optimizer copies
        # and overflow the user's private Kaggle output volume.
        # Keep existing per-step snapshots unchanged for all older paths.
        if h119_full_plan is not None:
            from h119.training_plan import checkpoint_due
        elif h122_grid_plan is not None:
            from h122.training_plan import checkpoint_due
        if (h119_full_plan is None and h122_grid_plan is None) or checkpoint_due(step+1,config.max_steps):
            version = publish_checkpoint(output, progress, write_version)
            interrupt_after_checkpoint(stop_after_step, step + 1, version)
    result = completed_checkpoint_result(version, progress, config_data)
    atomic_json(output / 'training-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'examples', 'manifest', 'baseline', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init-adapter', default=None)
    args = parser.parse_args()
    train(args.config, args.examples, args.manifest, args.baseline, args.output, args.resume,
          init_adapter=args.init_adapter)


if __name__ == '__main__':
    main()
