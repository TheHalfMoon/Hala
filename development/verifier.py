"""Final-state checks deliberately independent of simulator/oracle/parser imports."""
import re


def verify_mind_semantics(goal, observation, response, requested_kind):
    """Score decomposition using public user text/state, never the oracle policy.

    This checker is specific to the authored bilingual replace/move/save tasks.
    It supplies no hints or expected subgoals to the model or Type executor.
    """
    if response.get('kind') != requested_kind:
        return False
    quoted = [left or right for left, right in re.findall(r'"([^\"]+)"|«([^»]+)»', goal)]
    if len(quoted) != 2:
        return False
    requested_value, destination = quoted
    if requested_kind == 'plan':
        expected = [{'tool': 'visual', 'subgoal': name} for name in
                    ('replace_field', 'move_card', 'save_changes')]
        expected.append({'tool': 'observe', 'subgoal': 'verify_save'})
        return response.get('steps') == expected
    failure = ('temporarily failed' in observation['status'] or 'تعذر الحفظ' in observation['status'])
    if failure:
        tool, subgoal = 'observe', 'inspect_failure'
    elif observation['field_value'] != requested_value:
        tool, subgoal = 'visual', 'replace_field'
    elif observation['item_location'] != destination:
        tool, subgoal = 'visual', 'move_card'
    else:
        tool, subgoal = 'visual', 'save_changes'
    return response == {'kind': 'next', 'tool': tool, 'subgoal': subgoal}


def verify_final(episode, state):
    if not isinstance(state, dict):
        return False
    required = {'value', 'location', 'saved_value', 'saved_location', 'revision'}
    if not required <= state.keys():
        return False
    # The public goal requests a named value and destination; a real save must
    # persist both. A model's 'success' field is never consulted.
    return (state['value'] == episode['requested_value'] and
            state['location'] == episode['destination'] and
            state['saved_value'] == episode['requested_value'] and
            state['saved_location'] == episode['destination'] and
            type(state['revision']) is int and state['revision'] >= 1)
