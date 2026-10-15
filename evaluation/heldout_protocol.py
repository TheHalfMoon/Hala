"""H14 experimental coordinate advice; never an execution authority."""
import json
import math

REVISION = 'hala.h14.normalized-point.v1'
INSTRUCTION = ('Return only JSON: {"point":[x,y]}. Use integer normalized coordinates '
               'from 0 to 1000 on each image axis. The supplied image is the entire '
               'coordinate frame. Do not execute actions or claim completion.')


def dimensions(width, height):
    if any(type(v) is not int or not 1 <= v <= 4096 for v in (width, height)):
        raise ValueError('Integer image dimensions in 1..4096 required')


def parse(raw, width, height):
    dimensions(width, height)
    if type(raw) is not str or not 1 <= len(raw) <= 4096:
        raise ValueError('Bounded JSON text required')
    raw = raw.strip()
    # Explicit calibrated wrapper, not protocol guessing or arbitrary JSON extraction.
    if raw.startswith('```json\n') and raw.endswith('\n```'):
        raw = raw[8:-4]
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate key')
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('Nonfinite constant')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (RecursionError, json.JSONDecodeError) as error:
        raise ValueError('Malformed JSON') from error
    if (type(value) is not dict or set(value) != {'point'} or
            type(value['point']) is not list or len(value['point']) != 2 or
            any(type(v) is not int or not 0 <= v <= 1000 for v in value['point'])):
        raise ValueError('Exact integer point schema required')
    return tuple(min(size - 1, coordinate * size // 1000)
                 for coordinate, size in zip(value['point'], (width, height)))


def contains(box, point):
    left, top, right, bottom = box
    return left <= point[0] < right and top <= point[1] < bottom


def model_prompt(case):
    # Explicit allowlist: private target IDs, rectangles and fixture state never enter text.
    return case['goal'] + '\n' + INSTRUCTION


def score(case, raw):
    """Independent click fixture: apply a hit then inspect state; ignore model assertions."""
    try:
        point = parse(raw, *case['size'])
    except (ValueError, TypeError):
        return {'schema_valid': False, 'hit': False, 'wrong_target': False,
                'miss': False, 'fixture_complete': False, 'pixel_point': None}
    selected = next((c['id'] for c in case['controls'] if contains(c['box'], point)), None)
    # A real bounded state transition in this synthetic selection environment.
    state = {'selection': None, 'events': []}
    if selected is not None:
        state['selection'] = selected
        state['events'].append('selected')
    complete = (state['selection'] == case['expected_target'] and
                state['events'] == ['selected'])
    return {'schema_valid': True, 'hit': selected == case['expected_target'],
            'wrong_target': selected is not None and selected != case['expected_target'],
            'miss': selected is None, 'fixture_complete': complete, 'pixel_point': point}


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def summarize(rows):
    def group(items):
        n = len(items)
        return {'count': n, **{key: {'count': sum(bool(r[key]) for r in items),
                    'rate': sum(bool(r[key]) for r in items) / n if n else None,
                    'wilson_95_descriptive': wilson(sum(bool(r[key]) for r in items), n)}
                for key in ('schema_valid', 'hit', 'wrong_target', 'miss', 'fixture_complete')}}
    return {'overall': group(rows),
            'languages': {language: group([r for r in rows if r['language'] == language])
                          for language in ('en', 'ar')},
            'families': {family: group([r for r in rows if r['family'] == family])
                         for family in sorted({r['family'] for r in rows})},
            'uncertainty': 'Descriptive Wilson intervals only. Eight authored families have '
                           'dependent variants; neither instances nor families represent a '
                           'random sample of native desktop tasks.',
            'scope': 'Geometric grounding and one-click synthetic selection completion only; '
                     'native Windows/browser and general full-task qualification absent.'}
