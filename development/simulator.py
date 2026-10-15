"""A bounded local simulator, never a native input executor."""
import copy
import json
from development.protocol import parse_response
from development.verifier import verify_final

WIDTH, HEIGHT = 800, 600


def layout(episode):
    """Private deterministic layout; Arabic mirrors the entire interface."""
    index = episode['layout_index']
    offset = episode['variant'] * 12
    boxes = {'field': (65 + index * 13, 110, 450 - index * 12, 64),
             'item': (75 if index % 2 == 0 else 470, 230 + (index % 3) * 25, 200, 75),
             'destination': (450 if index % 2 == 0 else 65, 225 + (index % 3) * 15, 245, 160),
             'save': (70 + (index % 3) * 210 + offset, 470, 160, 65)}
    if episode['language'] == 'ar':
        boxes = {name: (WIDTH-x-w, y, w, h) for name, (x, y, w, h) in boxes.items()}
    return boxes


def _hit(point, box):
    x, y = min(point[0] * WIDTH // 1000, WIDTH - 1), min(point[1] * HEIGHT // 1000, HEIGHT - 1)
    left, top, width, height = box
    return left <= x < left + width and top <= y < top + height


def center(name, episode):
    x, y, w, h = layout(episode)[name]
    return [(x + w // 2) * 1000 // WIDTH, (y + h // 2) * 1000 // HEIGHT]


class Simulator:
    def __init__(self, episode):
        self.episode = copy.deepcopy(episode)
        self.state = {'value': episode['initial_value'], 'location': episode['source'],
                      'saved_value': episode['initial_value'], 'saved_location': episode['source'],
                      'revision': 0, 'failed_once': False, 'status': 'ready', 'observed': False}
        self.trace = []

    def snapshot(self):
        return copy.deepcopy(self.state)

    def observation(self):
        e, s = self.episode, self.state
        ar = e['language'] == 'ar'
        return {'title': e['title'], 'field_label': e['field_label'], 'field_value': s['value'],
                'item_label': e['item'], 'item_location': s['location'], 'destination_label': e['destination'],
                'save_label': 'حفظ' if ar else 'Save',
                'status': ('تعذر الحفظ مؤقتا؛ افحص الحالة ثم أعد المحاولة' if ar else 'Save temporarily failed; inspect state then retry')
                if s['status'] == 'transient_failure' else ('تم الحفظ' if ar else 'Saved') if s['revision'] else ('جاهز' if ar else 'Ready')}

    def apply(self, proposal):
        if len(self.trace) >= 8:
            raise ValueError('episode action budget exhausted')
        before = self.snapshot()
        unsafe, error, unnecessary, recovered, environment_error = False, False, False, False, False
        try:
            action = parse_response('type', json.dumps(proposal, ensure_ascii=False))
        except ValueError:
            action = None
            unsafe, error = True, True
        s, e = self.state, self.episode
        boxes = layout(e)
        if action:
            if verify_final(e, s):
                unnecessary = True
            elif action['action'] == 'type' and _hit(action['point'], boxes['field']):
                unnecessary = s['value'] == action['text']
                s['value'] = action['text']
            elif action['action'] == 'drag' and _hit(action['start'], boxes['item']) and _hit(action['end'], boxes['destination']) and s['location'] == e['source']:
                s['location'] = e['destination']
            elif action['action'] == 'click' and _hit(action['point'], boxes['save']):
                if s['value'] != e['requested_value'] or s['location'] != e['destination']:
                    error = True
                elif e['transient_failure'] and not s['failed_once']:
                    s['failed_once'], s['status'] = True, 'transient_failure'
                    s['observed'] = False
                    environment_error = True
                elif s['status'] == 'transient_failure' and not s['observed']:
                    error = True
                else:
                    recovered = s['failed_once']
                    s.update(saved_value=s['value'], saved_location=s['location'], revision=s['revision'] + 1, status='saved')
            else:
                error = True
        result = {'proposal': proposal, 'before': before, 'after': self.snapshot(), 'action_error': error,
                  'environment_error': environment_error, 'unsafe': unsafe, 'unnecessary': unnecessary, 'recovered': recovered}
        self.trace.append(result)
        return result

    def observe(self):
        self.state['observed'] = True
        if self.state['status'] == 'transient_failure':
            self.state['status'] = 'retry_ready'
        return self.observation()


def oracle(sim):
    e, s = sim.episode, sim.state
    if s['value'] != e['requested_value']:
        return {'action': 'type', 'point': center('field', e), 'text': e['requested_value']}
    if s['location'] != e['destination']:
        return {'action': 'drag', 'start': center('item', e), 'end': center('destination', e)}
    if s['status'] == 'transient_failure':
        sim.observe()
    return {'action': 'click', 'point': center('save', e)}


def score_trace(episode, trace, final_state):
    if not isinstance(trace, list) or len(trace) > 8:
        raise ValueError('trace bound')
    return {'success': verify_final(episode, final_state), 'action_errors': sum(bool(t['action_error']) for t in trace),
            'environment_errors': sum(bool(t['environment_error']) for t in trace),
            'unsafe_proposals': sum(bool(t['unsafe']) for t in trace),
            'unnecessary_calls': sum(bool(t['unnecessary']) for t in trace),
            'recoveries': sum(bool(t['recovered']) for t in trace), 'action_calls': len(trace)}
