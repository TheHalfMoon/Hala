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
from training.checkpoints import (IDENTITY_FIELDS, atomic_json, completed_checkpoint_result,
                                  load_checkpoint, publish_checkpoint)
from training.masking import completion_labels
from training.preflight import TrainingConfig, qualify_environment, validate_experiment
from training.scripts.gpu_probe import probe
from training.scripts.baseline import reject_kaggle_runtime

LORA_TARGET_PATTERN = r'.*language_model.*\.(q_proj|v_proj)$'


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Bounded experiment JSON required')
    return strict_json(path.read_text(encoding='utf-8'))


def prepare_inputs(row, processor, root, token_budget):
    """One bounded image; validate expanded vision/prompt prefix before GPU use."""
    import torch
    from PIL import Image
    image_path = artifact_path(root, row['image_path'])
    if sha256_file(image_path) != row['image_sha256']:
        raise ValueError('Training media digest changed after preflight')
    with Image.open(image_path) as source:
        if source.format != 'PNG' or source.width * source.height > MAX_PIXELS:
            raise ValueError('Training image exceeds header/pixel budget')
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
        return inputs, labels
    finally:
        image.close()


def train(config_path, examples_path, manifest_path, baseline_path, output_path, resume=False):
    if platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_TRAINING_ALLOWED') != '1':
        raise ValueError('Explicitly approved remote Linux runtime required; local weights forbidden')
    config_data = load_json(config_path)
    config = TrainingConfig.from_dict(config_data)
    if config.batch_size != 1:
        raise ValueError('Only candidate batch-size-one accumulation is implemented')
    if config.platform == 'kaggle':
        raise ValueError('Kaggle commercial-use eligibility is blocked by inspected provider terms')
    reject_kaggle_runtime()
    if config.model_id != 'Qwen/Qwen3-VL-2B-Instruct':
        raise ValueError('Only the candidate Qwen3-VL processor path is implemented')
    manifest = load_json(manifest_path)
    validate_experiment(config, manifest, load_json(baseline_path))
    records = load_sft_artifact(examples_path, manifest)
    root = Path(examples_path).parent.resolve()
    signature = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
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
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(42)
    dtype = torch.float16 if config.precision == 'fp16' else torch.bfloat16
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision,
                                             trust_remote_code=False)
    # Expanded vision tokens are included in this audit. No model weights allocated yet.
    for row in records:
        checked_inputs, checked_labels = prepare_inputs(row, processor, root, manifest['token_budget'])
        del checked_inputs, checked_labels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id, revision=config.revision, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation='sdpa').to('cuda:0')
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_rank * 2,
                      target_modules=LORA_TARGET_PATTERN,
                      lora_dropout=0, task_type='CAUSAL_LM', bias='none')
    model = (PeftModel.from_pretrained(model, version / 'adapter', is_trainable=True)
             if resume else get_peft_model(model, lora))
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
    scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'fp16')
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
    torch.cuda.reset_peak_memory_stats(0)
    for step in range(start_step, config.max_steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        losses = []
        for accumulation in range(config.gradient_accumulation):
            index = (step * config.gradient_accumulation + accumulation) % len(records)
            inputs, labels = prepare_inputs(records[index], processor, root, manifest['token_budget'])
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            with torch.autocast('cuda', dtype=dtype):
                loss = model(**inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss / config.gradient_accumulation).backward()
            losses.append(loss.item())
        scaler.unscale_(optimizer)
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in parameters):
            raise ValueError('Nonfinite adapter gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(0)
        current_changed = sum(not torch.equal(before, after.detach().cpu())
                              for before, after in zip(initial_parameters, parameters))
        changed = max(changed, current_changed)
        if not changed:
            raise ValueError('No verifiable adapter gradient update')
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
    result = completed_checkpoint_result(version, progress, config_data)
    atomic_json(output / 'training-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'examples', 'manifest', 'baseline', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.config, args.examples, args.manifest, args.baseline, args.output, args.resume)


if __name__ == '__main__':
    main()
