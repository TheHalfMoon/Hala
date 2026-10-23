"""Unexecuted remote-only Qwen3-VL LoRA candidate; no jobs or billing are created.

Pure artifact/mask/checkpoint helpers have CPU tests. Actual processor/training,
model fit and resume equivalence still require a pinned real remote experiment.
"""

import argparse
import hashlib
import json
import os
import platform
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
from training.scripts.gpu_probe import probe
from training.scripts.baseline import reject_kaggle_runtime

LORA_TARGET_PATTERN = r'.*language_model.*\.(q_proj|v_proj)$'


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
                   supervision='completion', coordinate_weight=8.0, return_weights=False):
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
          stop_after_step=None, init_adapter=None):
    if platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_TRAINING_ALLOWED') != '1':
        raise ValueError('Explicitly approved remote Linux runtime required; local weights forbidden')
    config_data = load_json(config_path)
    config = TrainingConfig.from_dict(config_data)
    if stop_after_step is not None and (type(stop_after_step) is not int or
                                       not 1 <= stop_after_step < config.max_steps):
        raise ValueError('Interruption must leave a remaining optimizer step')
    if config.batch_size != 1:
        raise ValueError('Only candidate batch-size-one accumulation is implemented')
    if config.platform == 'kaggle':
        raise ValueError('Kaggle commercial-use eligibility is blocked by inspected provider terms')
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
    native_drag = manifest.get('source_schema') in (H54_DRAG_SCHEMA, H57_DRAG_SCHEMA,
                                                   H60_DRAG_SCHEMA, H61_DRAG_SCHEMA)
    signature = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
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
    os.environ['HF_HOME'] = '/tmp/hala-model-cache'
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
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision,
                                             trust_remote_code=False)
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
    runtime['media_checked_records'] = len(records)
    runtime.update(audit_summary(audit_rows, len(records)))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id, revision=config.revision, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation='sdpa').to('cuda:0')
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_rank * 2,
                      target_modules=LORA_TARGET_PATTERN,
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
    if not trainable or any('lora_' not in name or 'visual' in name for name, _ in trainable):
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
            index = (step * config.gradient_accumulation + accumulation) % len(records)
            inputs, labels, weights = prepare_inputs(records[index], processor, root,
                                                    manifest['token_budget'],
                                                    native_drag_jpeg=native_drag,
                                                    supervision=config.supervision,
                                                    coordinate_weight=config.coordinate_weight,
                                                    return_weights=True)
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            weights = weights.to('cuda:0')
            microbatches.append({'record_id': records[index]['id'],
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
