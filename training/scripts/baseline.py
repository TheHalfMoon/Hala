"""Small unchanged-model GUI grounding smoke on approved remote Linux CUDA only.

Eight original cases are a diagnostic smoke, not the frozen Hala benchmark or release gate.
"""

import hashlib
import json
import os
import platform
import random
import signal
import time
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
MODEL_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
PROMPT_REVISION = 'hala-grounding-smoke-v1'


def parse_answer(answer, coordinate_mode='pixels'):
    if not isinstance(answer, str) or not 1 <= len(answer) <= 4096:
        raise ValueError('Bounded raw answer required')
    if coordinate_mode not in ('pixels', 'normalized_1000'):
        raise ValueError('Explicit coordinate protocol required')
    if coordinate_mode == 'normalized_1000':
        answer = answer.strip()
        if answer.startswith('```json\n') and answer.endswith('\n```'):
            answer = answer[8:-4]
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate output key')
            value[key] = item
        return value
    def constant(value):
        raise ValueError('Nonfinite output constant')
    try:
        value = json.loads(answer, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError as error:
        raise ValueError('Output nesting exceeds budget') from error
    if coordinate_mode == 'normalized_1000':
        if (not isinstance(value, dict) or set(value) != {'point'} or
                not isinstance(value['point'], list) or len(value['point']) != 2 or
                any(type(x) is not int or not 0 <= x <= 1000 for x in value['point'])):
            raise ValueError('Exact normalized point schema required')
        value = {'point': [min(639, value['point'][0] * 640 // 1000),
                           min(359, value['point'][1] * 360 // 1000)]}
    if not valid_point(value):
        raise ValueError('Exact point output schema required')
    return value


def valid_point(value):
    if not isinstance(value, dict) or set(value) != {'point'}:
        return False
    point = value['point']
    if not isinstance(point, list) or len(point) != 2:
        return False
    if any(type(x) is not int for x in point):
        return False
    return 0 <= point[0] < 640 and 0 <= point[1] < 360


def check_point(value, box):
    if not valid_point(value):
        return False
    point = value['point']
    left, top, right, bottom = box
    return left <= point[0] < right and top <= point[1] < bottom


def reject_kaggle_runtime():
    # Colab images can contain /kaggle without being Kaggle notebooks.
    # Provider markers are deployment diagnostics, not an adversarial boundary.
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or (
            Path('/kaggle').exists() and not os.environ.get('COLAB_RELEASE_TAG')):
        raise ValueError('Kaggle commercial-use eligibility is blocked')


def run(output_path='hala-baseline-result.json', coordinate_mode='pixels'):
    if coordinate_mode not in ('pixels', 'normalized_1000'):
        raise ValueError('Explicit coordinate protocol required')
    if platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_BASELINE_ALLOWED') != '1':
        raise ValueError('Explicitly approved remote Linux notebook required; no local weights')
    reject_kaggle_runtime()
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['HF_HOME'] = '/tmp/hala-model-cache'
    import torch
    import transformers
    from PIL import Image, ImageDraw, ImageFont, features
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise ValueError('No actual CUDA GPU available')
    if transformers.__version__ != '5.16.1':
        raise ValueError('Install the inspected transformers==5.16.1 before running')
    def timed_out(signum, frame):
        raise TimeoutError('Experiment exceeded 20-minute budget')
    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(1200)
    result = {'schema': 'hala.baseline.smoke.v1', 'model_id': MODEL_ID,
              'revision': MODEL_REVISION, 'prompt_revision': (PROMPT_REVISION if coordinate_mode == 'pixels'
                                                            else 'hala-grounding-calibration-v2'),
              'coordinate_protocol': coordinate_mode,
              'calibration_only': coordinate_mode != 'pixels',
              'evidence': 'real_inference', 'training_runs': 0, 'precision': 'fp16',
              'attention': 'sdpa', 'python': platform.python_version(),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'gpu': torch.cuda.get_device_name(0),
              'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
              'cases': [], 'release_qualified': False}
    model = None
    try:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                 trust_remote_code=False)
        torch.cuda.synchronize(0)
        result['download_and_load_seconds'] = time.perf_counter() - start
        result['loaded_allocated_bytes'] = torch.cuda.memory_allocated(0)
        raqm = features.check('raqm')
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        labels = {'en': ['Save', 'Cancel', 'Search', 'Open'],
                  'ar': ['حفظ', 'إلغاء', 'بحث', 'فتح']}
        for index in range(8):
            language = 'en' if index < 4 else 'ar'
            if language == 'ar' and not raqm:
                result['cases'].append({'id': index, 'language': language,
                                        'status': 'blocked', 'reason': 'RTL shaping unavailable'})
                continue
            choices = labels[language].copy()
            random.Random(index + 42).shuffle(choices)
            image = Image.new('RGB', (640, 360), '#eeeeee')
            draw = ImageDraw.Draw(image)
            draw.text((24, 20), 'Hala synthetic control fixture', font=font, fill='black')
            boxes = []
            for slot, label in enumerate(choices):
                x, y = 40 + (slot % 2) * 300, 100 + (slot // 2) * 100
                box = (x, y, x + 220, y + 60)
                boxes.append(box)
                draw.rectangle(box, fill='white', outline='#222222', width=2)
                draw.text((x + 110, y + 30), label, font=font, fill='black', anchor='mm',
                          **({'direction': 'rtl', 'language': 'ar'} if language == 'ar' else {}))
            target = labels[language][index % 4]
            goal = f'Click the {target} button.' if language == 'en' else f'انقر على زر {target}.'
            prompt = goal + '\nReturn only JSON: {"point":[x,y]}. Coordinates are integer pixels in the 640x360 image. Do not execute anything.'
            if coordinate_mode == 'normalized_1000':
                prompt = goal + '\nReturn only JSON: {"point":[x,y]}. Use integer normalized coordinates from 0 to 1000 on each image axis. Do not execute anything.'
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[template], images=[image], return_tensors='pt').to('cuda:0')
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=80, do_sample=False, max_time=45)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            try:
                value = parse_answer(answer, coordinate_mode)
                valid = valid_point(value)
                passed = check_point(value, boxes[choices.index(target)])
            except (ValueError, TypeError):
                valid, passed, value = False, False, None
            result['cases'].append({'id': index, 'language': language, 'goal': goal,
                                    'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                    'expected_box': boxes[choices.index(target)], 'output': answer,
                                    'schema_valid': valid, 'verified_target_hit': passed,
                                    'decoded_pixel_point': value,
                                    'inference_seconds': elapsed,
                                    'output_tokens': generated.shape[1] - inputs.input_ids.shape[1],
                                    'status': 'tested', 'state_verification_scope': 'geometric_target_only'})
            print(json.dumps(result['cases'][-1], ensure_ascii=False), flush=True)
        tested = [row for row in result['cases'] if row['status'] == 'tested']
        result['evaluated_tasks'] = len(tested)
        result['verified_target_hit_rate'] = sum(row['verified_target_hit'] for row in tested) / len(tested) if tested else None
        result['peak_vram_bytes'] = torch.cuda.max_memory_allocated(0)
        result['status'] = 'tested'
        result['limits'] = 'Eight diagnostic GUI cases only. No native action execution, complete-task success, quantization comparison or general bilingual qualification.'
    except Exception as error:
        result['status'] = 'blocked'
        result['error_type'] = type(error).__name__
        # Synthetic diagnostics only; avoid printing arbitrary exception payloads.
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, ensure_ascii=False))
