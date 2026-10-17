"""Fixed CPU subprocess fixture. Never imported by the operator loader."""
import json
import sys
import time
from hala.serving.lifecycle import PROVENANCE

def emit(value):
    print(json.dumps(value), flush=True)

if __name__ == '__main__':
    emit({'ready': PROVENANCE})
    for line in sys.stdin.buffer:
        value = json.loads(line)
        if value == {'probe':'allocator_budget'}:
            emit({'probe_result':{'kind':'allocator_budget','exhausted':True,'requested_bytes':8388608}})
            continue
        goal = value['request']['goal']
        if goal == 'wait':
            emit({'event':'cpu_work_started'})
            time.sleep(5)
        if goal == 'crash': sys.exit(4)
        if goal == 'oversize':
            print('x'*1100000, flush=True)
            continue
        if goal == 'resource': emit({'error':'busy'})
        else: emit({'output':'{"point":[500,500]}'})
