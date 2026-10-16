"""Independent persisted outcome check; never trusts service status or receipts."""
import json
from pathlib import Path


def verify_outcome(path, task):
    try:
        raw = Path(path).read_bytes()
        if len(raw)>4096: return False
        value = json.loads(raw)
        return (type(value) is dict and set(value) == {'task_id','settings'} and
                value['task_id'] == task['id'] and type(value['settings']) is dict and
                set(value['settings']) == {task['key']} and
                value['settings'][task['key']] is task['value'])
    except (OSError, ValueError, TypeError, KeyError):
        return False
