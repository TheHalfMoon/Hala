"""Small synchronous model seam; model proposals never execute native actions.

infer(component, prompt, image_path) -> bare JSON string. Mind uses no image;
Type uses the current screenshot. The callback sees no episode/private labels.
"""
import hashlib
import time
from pathlib import Path
from development.data import canonical, prompt
from development.media import render_frame
from development.protocol import parse_response
from development.simulator import Simulator, score_trace
from development.verifier import verify_final, verify_mind_semantics


def run_mind_cases(cases, infer):
    records = []
    for case in cases:
        start = time.perf_counter()
        record = {key: case[key] for key in ('id', 'language', 'split', 'category')}
        try:
            raw = infer('mind', case['prompt'], None)
            parsed = parse_response('mind', raw)
            record.update(raw=raw, valid=True, correct=parsed == case['expected'])
        except (ValueError, RuntimeError, TimeoutError) as exc:
            record.update(valid=False, correct=False, error=type(exc).__name__)
        record['seconds'] = time.perf_counter() - start
        records.append(record)
    return records


def run_episode(episode, infer, media_dir, font_path, max_calls=16):
    if type(max_calls) is not int or not 1 <= max_calls <= 16:
        raise ValueError('bounded model call budget required')
    sim, records, model_errors, redundant_observations = Simulator(episode), [], 0, 0
    started = time.perf_counter()
    for step in range(8):
        if verify_final(episode, sim.snapshot()) or len(records) >= max_calls:
            break
        requested_kind = 'plan' if step == 0 else 'next'
        observation = sim.observation()
        mind_prompt = prompt('mind', episode, observation, requested_kind)
        record = {'component': 'mind', 'prompt': mind_prompt, 'requested_kind': requested_kind,
                  'requested_kind_correct': False, 'semantic_correct': False}
        call_start = time.perf_counter()
        try:
            raw = infer('mind', mind_prompt, None)
            record['raw'] = raw
            mind = parse_response('mind', raw)
            record['requested_kind_correct'] = mind['kind'] == requested_kind
            record['semantic_correct'] = verify_mind_semantics(episode['goal'], observation, mind, requested_kind)
            if not record['requested_kind_correct']:
                raise ValueError('response kind differs from requested kind')
            tool = mind['steps'][0]['tool'] if mind['kind'] == 'plan' else mind['tool']
        except (ValueError, RuntimeError, TimeoutError) as exc:
            record['error'] = type(exc).__name__
            tool, model_errors = None, model_errors + 1
        record['seconds'] = time.perf_counter() - call_start
        records.append(record)
        if tool in ('clarify', 'abstain'):
            break
        if tool == 'observe':
            if sim.state['status'] != 'transient_failure':
                redundant_observations += 1
            sim.observe()
            continue
        if tool != 'visual' or len(records) >= max_calls:
            continue
        state = sim.snapshot()
        key = hashlib.sha256(canonical({'episode': episode['id'], 'state': state})).hexdigest()
        image_path = Path(media_dir) / (key + '.png')
        media = render_frame(episode, state, image_path, font_path)
        type_prompt = prompt('type', episode, sim.observation())
        record = {'component': 'type', 'prompt': type_prompt, 'image': str(image_path), 'media': media}
        call_start = time.perf_counter()
        try:
            raw = infer('type', type_prompt, str(image_path))
            record['raw'] = raw
            proposal = parse_response('type', raw)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            record['error'] = type(exc).__name__
            proposal, model_errors = {'action': 'invalid'}, model_errors + 1
        record['seconds'] = time.perf_counter() - call_start
        records.append(record)
        sim.apply(proposal)
    metrics = score_trace(episode, sim.trace, sim.snapshot())
    metrics['unnecessary_calls'] += redundant_observations
    metrics['redundant_observations'] = redundant_observations
    mind_calls = [r for r in records if r['component'] == 'mind']
    plan_calls = [r for r in mind_calls if r['requested_kind'] == 'plan']
    mind_correct = sum(r['semantic_correct'] for r in mind_calls)
    plan_correct = sum(r['semantic_correct'] for r in plan_calls)
    metrics.update(mind_calls=len(mind_calls), mind_correct=mind_correct,
                   mind_errors=len(mind_calls)-mind_correct, mind_accuracy=mind_correct/len(mind_calls) if mind_calls else None,
                   plan_calls=len(plan_calls), plan_correct=plan_correct,
                   plan_accuracy=plan_correct/len(plan_calls) if plan_calls else None)
    metrics.update(model_calls=len(records), model_errors=model_errors, wall_seconds=time.perf_counter()-started,
                   model_seconds=sum(r['seconds'] for r in records))
    return {'episode': episode['id'], 'family': episode['family'], 'language': episode['language'],
            'split': episode['split'], 'calls': records, 'trace': sim.trace,
            'final_state': sim.snapshot(), 'metrics': metrics, 'native_qualified': False}
