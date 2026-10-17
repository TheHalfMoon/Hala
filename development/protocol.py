"""Versioned untrusted proposals, with no authority-bearing fields."""
import json

VERSION = 'hala.development.v1'
TOOLS = {'visual', 'observe', 'clarify', 'abstain'}
SUBGOALS = {'visual': {'replace_field', 'move_card', 'save_changes'},
            'observe': {'inspect_failure', 'verify_save'}, 'clarify': {'resolve_destination'},
            'abstain': {'reject_external_command'}}


def _step(value):
    return (isinstance(value, dict) and set(value) == {'tool', 'subgoal'} and
            isinstance(value.get('tool'), str) and value['tool'] in SUBGOALS and
            isinstance(value.get('subgoal'), str) and value['subgoal'] in SUBGOALS[value['tool']])


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate field')
        result[key] = value
    return result


def parse_response(component, text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 4096:
        raise ValueError('response bound')
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    except (TypeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('bare JSON required') from exc
    if not isinstance(value, dict):
        raise ValueError('object required')
    if component == 'type':
        action = value.get('action')
        fields = {'click': {'action', 'point'}, 'type': {'action', 'point', 'text'},
                  'drag': {'action', 'start', 'end'}}
        if not isinstance(action, str) or action not in fields or set(value) != fields[action]:
            raise ValueError('unknown action or fields')
        for name in ('point', 'start', 'end'):
            if name in value and (not isinstance(value[name], list) or len(value[name]) != 2 or
                                any(type(n) is not int or not 0 <= n <= 1000 for n in value[name])):
                raise ValueError('normalized integer point required')
        if action == 'type' and (not isinstance(value['text'], str) or not 1 <= len(value['text']) <= 80 or
                                 any(ord(c) < 32 for c in value['text'])):
            raise ValueError('bounded plain text required')
    elif component == 'mind':
        kind = value.get('kind')
        if kind == 'plan':
            if set(value) != {'kind', 'steps'} or not isinstance(value['steps'], list) or not 1 <= len(value['steps']) <= 4 or any(not _step(s) for s in value['steps']):
                raise ValueError('bounded tool plan required')
        elif kind == 'next':
            if set(value) != {'kind', 'tool', 'subgoal'} or not _step({k: value[k] for k in ('tool', 'subgoal')}):
                raise ValueError('bounded next tool required')
        else:
            raise ValueError('unknown Mind kind')
    elif component == 'reliance':
        if set(value) != {'route'} or not isinstance(value['route'], str) or value['route'] not in {'deterministic', 'mind', 'type', 'observe', 'clarify', 'abstain'}:
            raise ValueError('advisory route required')
    else:
        raise ValueError('unknown component')
    return value


def instruction(component):
    common = 'Return one bare JSON object only. Proposals carry no authority. '
    if component == 'type':
        return common + ('Use screenshot coordinates normalized to integers 0..1000. Allowed: '
                         '{"action":"click","point":[x,y]}, '
                         '{"action":"type","point":[x,y],"text":"value"} (replace field), '
                         '{"action":"drag","start":[x,y],"end":[x,y]}. Propose the next single action toward the goal. Do not repeat completed actions. ')
    if component == 'mind':
        return common + ('Allowed tools: visual, observe, clarify, abstain. Return '
                         '{"kind":"plan","steps":[{"tool":"visual","subgoal":"replace_field"},...]} when asked for a plan, otherwise '
                         '{"kind":"next","tool":"visual","subgoal":"replace_field"}. '
                         'Allowed subgoals: visual=replace_field|move_card|save_changes; observe=inspect_failure|verify_save; '
                         'clarify=resolve_destination; abstain=reject_external_command. Missing essential user intent requires clarify; forbidden external actions require abstain; a transient failure requires observe. ')
    return common + 'Return {"route":"deterministic|mind|type|observe|clarify|abstain"}; advisory only. '
