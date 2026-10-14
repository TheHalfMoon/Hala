"""Optional pinned remote GPU engine. Never import ML dependencies on CPU CI."""
import os
import platform
from pathlib import Path
import threading
import time

from hala.inference import ServiceError
from hala.inference.vision_candidate import PROMPT_REVISION, INSTRUCTION

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
MODEL_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
TORCH_VERSION = '2.11.0'
TRANSFORMERS_VERSION = '5.16.1'


class RemoteQwenEngine:
    """Source candidate, not a runtime qualification or deployment entrypoint.

    Loading is restricted to explicitly opted-in Linux Colab ephemeral roots. The caller must
    verify a free GPU session before calling this optional loader. No startup
    path invokes it automatically and there is no caller qualification flag.
    """
    provenance = {'engine': 'remote_transformers', 'model_id': MODEL_ID,
                  'model_revision': MODEL_REVISION, 'processor_revision': MODEL_REVISION,
                  'precision': 'float16', 'prompt_revision': PROMPT_REVISION}

    @classmethod
    def load(cls, cache_dir):
        path = Path(cache_dir).resolve()
        if platform.system() != 'Linux' or os.environ.get('HALA_REMOTE_SERVING_ALLOWED') != '1':
            raise ValueError('Explicit remote Linux serving opt-in required; no local weights.')
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or (Path('/kaggle').exists() and not os.environ.get('COLAB_RELEASE_TAG')):
            raise ValueError('Kaggle commercial-use eligibility is blocked.')
        if not os.environ.get('COLAB_RELEASE_TAG'):
            raise ValueError('Actual Colab runtime marker required.')
        roots = (Path('/content'), Path('/tmp/hala-serving-model-cache'))
        if not any(path == root or path.is_relative_to(root) for root in roots):
            raise ValueError('Colab /content or approved /tmp serving cache required.')
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'
        import torch
        import transformers
        if torch.__version__.split('+')[0] != TORCH_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError('Pinned Torch/Transformers runtime required.')
        if not torch.cuda.is_available():
            raise ValueError('Remote CUDA GPU required.')
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                    trust_remote_code=False, cache_dir=str(path), min_pixels=256*28*28,
                    max_pixels=1024*28*28)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
                    revision=MODEL_REVISION, trust_remote_code=False, cache_dir=str(path),
                    dtype=torch.float16, attn_implementation='eager')
        model.to('cuda:0')
        model.eval()
        engine = cls()
        engine.processor, engine.model, engine.torch = processor, model, torch
        engine.lock = threading.Lock()
        return engine

    def __call__(self, image, request, context):
        from transformers import StoppingCriteria, StoppingCriteriaList
        context.check()
        if not self.lock.acquire(blocking=False):
            raise ServiceError('busy')
        try:
            class Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return context.cancelled.is_set() or time.monotonic() >= context.deadline
            prompt = request['goal'] + '\n' + INSTRUCTION
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                     {'type': 'text', 'text': prompt}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors='pt')
            if inputs['input_ids'].shape[1] > 4096:
                raise ServiceError('invalid_request')
            inputs = inputs.to(self.model.device)
            context.check()
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=96,
                    stopping_criteria=StoppingCriteriaList([Stop()]),
                    max_time=max(.001, context.deadline-time.monotonic()))
            context.check()
            text = self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if len(text.encode('utf-8')) > 2048:
                raise ServiceError('invalid_backend_output')
            return text
        finally:
            self.lock.release()
