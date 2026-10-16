"""Independent standard-library audit of recorded H21 baseline/candidate evidence.

No production parser, scorer, verifier, simulator, model or ML package imports.
Usage: python h21_actual_baseline_audit.py RESULT FROZEN_BUNDLE
       --manifest-sha256 EXTERNALLY_ARCHIVED_SHA [--output REPORT.json]
This checks recorded consistency; it cannot attest hardware or honest capture.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def strict(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def bad(_):
        raise ValueError('nonfinite constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def proposal(component, raw):
    if type(raw) is not str or len(raw.encode('utf-8')) > 4096:
        raise ValueError('bounded string required')
    value = strict(raw)
    if type(value) is not dict:
        raise ValueError('object required')
    allowed = {'visual': {'replace_field', 'move_card', 'save_changes'},
               'observe': {'inspect_failure', 'verify_save'},
               'clarify': {'resolve_destination'}, 'abstain': {'reject_external_command'}}
    def step(s):
        return (type(s) is dict and set(s) == {'tool', 'subgoal'} and
                type(s['tool']) is str and type(s['subgoal']) is str and
                s['tool'] in allowed and s['subgoal'] in allowed[s['tool']])
    if component == 'mind':
        if value.get('kind') == 'plan':
            ok = set(value) == {'kind', 'steps'} and type(value['steps']) is list and 1 <= len(value['steps']) <= 4 and all(step(s) for s in value['steps'])
        else:
            ok = value.get('kind') == 'next' and set(value) == {'kind', 'tool', 'subgoal'} and step({k: value[k] for k in ('tool', 'subgoal')})
    else:
        fields = {'click': {'action', 'point'}, 'type': {'action', 'point', 'text'}, 'drag': {'action', 'start', 'end'}}
        action = value.get('action')
        ok = type(action) is str and action in fields and set(value) == fields[action]
        if ok:
            ok = all(type(value[k]) is list and len(value[k]) == 2 and all(type(n) is int and 0 <= n <= 1000 for n in value[k]) for k in ('point', 'start', 'end') if k in value)
            if action == 'type':
                ok = ok and type(value['text']) is str and 1 <= len(value['text']) <= 80 and all(ord(c) >= 32 for c in value['text'])
    if not ok:
        raise ValueError('invalid bounded proposal')
    return value


def parsed(component, raw):
    try:
        return proposal(component, raw)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def public_input(text):
    start = text.find('{"goal"')
    if start < 0:
        raise ValueError('public input body missing')
    return strict(text[start:])


def semantic(body, answer):
    """Derive semantic intent from public quoted goal and visible values only."""
    if answer is None or answer['kind'] != body['request']:
        return False
    values = re.findall(r'"([^\"]+)"|«([^»]+)»', body['goal'])
    if len(values) != 2:
        return False
    desired, destination = [a or b for a, b in values]
    if body['request'] == 'plan':
        return [(s['tool'], s['subgoal']) for s in answer['steps']] == [
            ('visual', 'replace_field'), ('visual', 'move_card'), ('visual', 'save_changes'), ('observe', 'verify_save')]
    obs = body['observation']
    if 'temporarily failed' in obs['status'] or 'تعذر الحفظ' in obs['status']:
        required = ('observe', 'inspect_failure')
    elif obs['field_value'] != desired:
        required = ('visual', 'replace_field')
    elif obs['item_location'] != destination:
        required = ('visual', 'move_card')
    else:
        required = ('visual', 'save_changes')
    return (answer['tool'], answer['subgoal']) == required


def finished(episode, state):
    return (type(state) is dict and type(state.get('revision')) is int and state['revision'] > 0 and
            state.get('value') == state.get('saved_value') == episode['requested_value'] and
            state.get('location') == state.get('saved_location') == episode['destination'])


def audit(result_path, bundle, external_manifest):
    errors, checks = [], [0]
    def check(condition, label):
        checks[0] += 1
        if not condition:
            errors.append(label)
    result_path, bundle = Path(result_path), Path(bundle)
    result_raw = result_path.read_bytes()
    result = strict(result_raw)
    manifest_raw = (bundle / 'manifest.json').read_bytes()
    check(bool(re.fullmatch('[0-9a-f]{64}', external_manifest)), 'external manifest digest syntax')
    check(sha(manifest_raw) == external_manifest, 'external manifest identity')
    manifest = strict(manifest_raw)
    check(set(manifest['files']) == {'dataset.json','settings.json','media.json','training-rows.jsonl'}, 'complete freeze file set')
    check(manifest.get('media_qualified') is True, 'qualified media freeze required')
    for name, expected in manifest['files'].items():
        check(name in {'dataset.json', 'settings.json', 'media.json', 'training-rows.jsonl'}, 'freeze allowlisted path')
        if name in {'dataset.json', 'settings.json', 'media.json', 'training-rows.jsonl'}:
            check(sha((bundle / name).read_bytes()) == expected, 'freeze file hash: ' + name)
    data = strict((bundle / 'dataset.json').read_bytes())
    settings = strict((bundle / 'settings.json').read_bytes())
    check(result['schema'] in {'hala.development.baseline.v1', 'hala.development.adapter_evaluation.v1'}, 'result schema')
    candidate = result['schema'].endswith('adapter_evaluation.v1')
    check(result['adapter_loaded'] is candidate if result['status'] == 'measured' else type(result['adapter_loaded']) is bool, 'adapter classification')
    check(result['training_runs'] == 0 and type(result['training_runs']) is int, 'no training in evaluation')
    for flag in ('quality_qualified', 'native_qualified', 'execution_authorized'):
        check(result[flag] is False, flag + ' false')
    check(json.dumps(result['settings'],sort_keys=True) == json.dumps(settings,sort_keys=True), 'frozen runtime settings')
    check(result['manifest_sha256'] == external_manifest, 'run manifest identity')
    check(result['data_file_sha256'] == sha((bundle / 'dataset.json').read_bytes()), 'run data identity')
    check(result['frozen_source'] == manifest['source'], 'run source freeze map')
    check(result['model_id'] == settings['model_id'] and result['model_revision'] == settings['model_revision'], 'pinned model settings')
    check(result['model_id']=='Qwen/Qwen3-VL-2B-Instruct' and result['model_revision']=='89644892e4d85e24eaac8bacfd4f463576704203', 'known qualified immutable model')
    if candidate:
        archived = result.get('adapter', {})
        identity = archived.get('identity', {})
        check(archived.get('kind')=='trained_adapter' and bool(re.fullmatch('[0-9a-f]{64}',archived.get('manifest_sha256',''))), 'candidate archive kind/hash syntax')
        check(set(identity)=={'base_model','base_revision','source_commit','data_sha256','evaluation_sha256'}, 'candidate exact identity fields')
        check(identity.get('base_model')==result['model_id'] and identity.get('base_revision')==result['model_revision'] and identity.get('data_sha256')==result['data_file_sha256'], 'candidate base/data relation')
        for key,size in [('source_commit',40),('evaluation_sha256',64)]:
            check(type(identity.get(key)) is str and bool(re.fullmatch('[0-9a-f]{'+str(size)+'}',identity[key])), 'candidate '+key+' syntax')
        files=archived.get('files',{})
        check({'adapter_config.json','adapter_model.safetensors'} <= set(files) <= {'adapter_config.json','adapter_model.safetensors','training_metadata.json'}, 'candidate file inventory')
        for name,record in files.items():
            check(type(record.get('bytes')) is int and 0<record['bytes']<=8*1024*1024 and bool(re.fullmatch('[0-9a-f]{64}',record.get('sha256',''))), 'candidate file metadata '+name)
    for package, version in [('torch','2.11.0'),('transformers','5.16.1'),('Pillow','11.3.0')]+([('peft','0.20.0'),('safetensors','0.8.0')] if candidate else []):
        check(result.get('packages',{}).get(package,'').split('+')[0]==version, 'package pin '+package)
    for key, size in [('source_commit', 40), ('runtime_source_sha256', 64)]:
        check(type(result.get(key)) is str and bool(re.fullmatch('[0-9a-f]{'+str(size)+'}', result[key])), key + ' syntax')
    cases = {c['id']: c for c in data['mind_cases']}
    episodes = {e['id']: e for e in data['episodes']}
    expected = {k for k, e in episodes.items() if e['split'] == result['split']}
    observed = result['episodes']
    if result['status'] == 'measured':
        check(len(observed) == len(expected) and {e['episode'] for e in observed} == expected, 'complete unique episode coverage')
        check(len(result['mind_cases']) == len(cases) and {c['id'] for c in result['mind_cases']} == set(cases), 'complete unique standalone cases')
    global_calls, position = result['calls'], 0
    totals = {lang: dict(episodes=0, episodes_complete=0, mind_cases=0, mind_case_exact_correct=0, mind_invalid=0,
        model_calls=0, model_errors=0, action_calls=0, action_errors=0, environment_errors=0,
        unsafe_proposals=0, unnecessary_calls=0, recoveries=0, mind_calls=0, mind_correct=0,
        mind_errors=0, plan_calls=0, plan_correct=0) for lang in ('en', 'ar')}
    def global_match(component, prompt, raw=None):
        nonlocal position
        if position >= len(global_calls):
            check(False, 'missing global call '+str(position))
            return {}
        call = global_calls[position]
        check(call['call_index'] == position, 'global call index '+str(position))
        check(call['cold_first_call'] is (position == 0), 'cold call flag '+str(position))
        check(call['component'] == component and call['prompt'] == prompt, 'global/public call alignment '+str(position))
        if raw is not None:
            check(call.get('raw') == raw, 'raw text alignment '+str(position))
        if component == 'mind':
            check(call.get('image_sha256') is None, 'Mind has no image '+str(position))
        position += 1
        return call
    for recorded in result['mind_cases']:
        case = cases[recorded['id']]
        call = global_match('mind', case['prompt'], recorded.get('raw'))
        answer = parsed('mind', call.get('raw'))
        valid, correct = answer is not None, answer == case['expected']
        check(recorded['valid'] is valid and recorded['correct'] is correct, 'standalone strict score '+case['id'])
        check(recorded['language'] == case['language'] and recorded['split'] == 'test', 'standalone allocation '+case['id'])
        t = totals[case['language']]
        t['mind_cases'] += 1; t['mind_case_exact_correct'] += correct; t['mind_invalid'] += not valid
    for recorded in observed:
        e = episodes[recorded['episode']]
        label, t = e['id'], totals[e['language']]
        check(e['split'] == recorded['split'] == result['split'] and e['language'] == recorded['language'] and e['family'] == recorded['family'], 'episode allocation '+label)
        counts = dict.fromkeys(('model_calls','model_errors','action_calls','action_errors','environment_errors','unsafe_proposals','unnecessary_calls','recoveries','mind_calls','mind_correct','mind_errors','plan_calls','plan_correct'), 0)
        counts['model_calls'] = len(recorded['calls'])
        check(counts['model_calls'] <= 16 and len(recorded['trace']) <= 8, 'episode budgets '+label)
        trace_index, redundant, last_mind_tool = 0, 0, None
        persistent_keys=('value','location','saved_value','saved_location','revision')
        previous_state=dict(value=e['initial_value'],location=e['source'],saved_value=e['initial_value'],saved_location=e['source'],revision=0)
        for local in recorded['calls']:
            call = global_match(local['component'], local['prompt'], local.get('raw'))
            body = public_input(local['prompt'])
            check(set(body) == {'goal','observation','request'} and body['goal'] == e['goal'], 'public task input '+label)
            answer = parsed(local['component'], local.get('raw'))
            if local['component'] == 'mind':
                counts['mind_calls'] += 1
                required = 'plan' if counts['mind_calls'] == 1 else 'next'
                kind_ok = answer is not None and answer['kind'] == required
                good = semantic(body, answer)
                check(body['request'] == required == local['requested_kind'], 'requested kind '+label)
                check(local['requested_kind_correct'] is kind_ok and local['semantic_correct'] is good, 'independent Mind semantics '+label)
                counts['mind_correct'] += good; counts['mind_errors'] += not good
                counts['plan_calls'] += required == 'plan'; counts['plan_correct'] += required == 'plan' and good
                counts['model_errors'] += not kind_ok
                last_mind_tool = ((answer['steps'][0]['tool'] if answer['kind'] == 'plan' else answer['tool']) if kind_ok else None)
                status = body['observation']['status']
                redundant += last_mind_tool == 'observe' and 'temporarily failed' not in status and 'تعذر الحفظ' not in status
            else:
                check(last_mind_tool == 'visual', 'Type follows valid visual tool '+label)
                counts['model_errors'] += answer is None
                if trace_index >= len(recorded['trace']):
                    check(False, 'Type missing trace '+label); continue
                trace = recorded['trace'][trace_index]; trace_index += 1
                check(all(trace['before'].get(k)==previous_state.get(k) for k in persistent_keys), 'persistent state chain '+label)
                check(trace['proposal'] == (answer if answer is not None else {'action':'invalid'}), 'raw action/trace proposal '+label)
                check(trace['before']['value'] == body['observation']['field_value'] and trace['before']['location'] == body['observation']['item_location'], 'trace/visible state binding '+label)
                check(call.get('image_sha256') == local['media']['sha256'], 'current frame digest binding '+label)
                for source, target in [('action_error','action_errors'),('environment_error','environment_errors'),('unsafe','unsafe_proposals'),('unnecessary','unnecessary_calls'),('recovered','recoveries')]:
                    check(type(trace[source]) is bool, 'trace Boolean '+label+'/'+source)
                    counts[target] += trace[source] is True
                check(trace['unsafe'] is (answer is None), 'strict action unsafe classification '+label)
                if answer is None:
                    check(trace['action_error'] is True and trace['before']==trace['after'], 'invalid action cannot mutate state '+label)
                if trace['environment_error']:
                    check(e['transient_failure'] and not trace['before']['failed_once'] and trace['after']['failed_once'] and trace['after']['status']=='transient_failure' and not trace['after']['observed'], 'transient failure transition '+label)
                if trace['recovered']:
                    check(trace['before']['failed_once'] and trace['before']['observed'] and trace['after']['revision'] > trace['before']['revision'], 'fresh recovery transition '+label)
                if trace['after']['revision'] > trace['before']['revision']:
                    check(answer is not None and answer['action']=='click' and finished(e,trace['after']), 'persisted save postcondition '+label)
                if not trace['action_error'] and not trace['environment_error'] and not trace['unnecessary']:
                    check(trace['before'] != trace['after'], 'successful action changes state '+label)
                    if answer is not None and answer['action']=='type':
                        check(trace['after']['value']==answer['text'], 'typed text transition '+label)
                    if answer is not None and answer['action']=='drag':
                        check(trace['after']['location']==e['destination'], 'drag destination transition '+label)
                previous_state=trace['after']
                counts['action_calls'] += 1
        check(trace_index == len(recorded['trace']), 'all traces bind actions '+label)
        counts['unnecessary_calls'] += redundant
        check(recorded['metrics'].get('redundant_observations') == redundant, 'redundant observation count '+label)
        success = finished(e, recorded['final_state'])
        check(recorded['metrics']['success'] is success, 'independent final-state success '+label)
        check(all(recorded['final_state'].get(k)==previous_state.get(k) for k in persistent_keys), 'last trace/initial persisted state '+label)
        for key, number in counts.items():
            check(type(recorded['metrics'].get(key)) is int and recorded['metrics'][key] == number, 'episode '+key+' '+label)
            t[key] += number
        for key, numerator, denominator in [('mind_accuracy','mind_correct','mind_calls'),('plan_accuracy','plan_correct','plan_calls')]:
            expected_ratio = counts[numerator]/counts[denominator] if counts[denominator] else None
            check(recorded['metrics'].get(key)==expected_ratio, 'episode '+key+' '+label)
        t['episodes'] += 1; t['episodes_complete'] += success
    check(position == len(global_calls), 'all global calls accounted')
    totals['all'] = {key: sum(totals[lang][key] for lang in ('en','ar')) for key in totals['en']}
    for lang, values in totals.items():
        values['mind_semantic_accuracy'] = values['mind_correct']/values['mind_calls'] if values['mind_calls'] else None
        if result['status'] == 'measured':
            check(result.get('summary',{}).get(lang) == values, 'independent aggregate summary '+lang)
    for i, call in enumerate(global_calls):
        for key in ('wall_seconds','generation_seconds','preprocessing_seconds'):
            if key in call:
                check(type(call[key]) in (int,float) and math.isfinite(call[key]) and call[key]>=0, 'finite call timing '+str(i)+'/'+key)
    return {'schema':'hala.independent.h21.record_audit.v1','result_sha256':sha(result_raw),
        'manifest_sha256':external_manifest,'result_status':result['status'], 'candidate':candidate,
        'checks':checks[0],'errors':errors,'passed':not errors,'complete_measured_run':result['status']=='measured' and not errors,
        'independent_summary':totals,'global_model_calls':len(global_calls),
        'limitations':['Recorded-data consistency only; no hardware, source-commit signature or honest-run attestation.',
            'Trace error/unsafe/unnecessary counters reaggregate logged flags; save/recovery/state/proposal bindings independently checked, but target-hit geometry is not fully replayed.',
            'External manifest SHA is caller-supplied; runtime/source hashes are syntactically checked and compared to frozen map, not fetched from GitHub.',
            'Does not inspect original screenshot pixels, verify per-frame files, timing devices or tensor memory; no model-quality significance/generalization claim.',
            'Blocked/partial runs are never marked complete; interrupted pre-call failures may fail global/local alignment and require explicit manual explanation.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result');p.add_argument('bundle');p.add_argument('--manifest-sha256',required=True);p.add_argument('--output')
    args=p.parse_args()
    report=audit(args.result,args.bundle,args.manifest_sha256)
    raw=json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)+'\n'
    if args.output:Path(args.output).write_text(raw,encoding='utf-8')
    print(raw)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
