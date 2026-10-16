"""Private H23 synthetic operator. Existing v2 proposals only, no native execution."""
import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import secrets
import threading
import time
import socket
import importlib.metadata
import subprocess

from hala.inference import ServiceError
from hala.serving import Config, Service, create_server, encode, strict_json
from hala.inference.vision_candidate import CandidateBackend
from evaluation.serving_lifecycle_verifier import verify_outcome

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'experiments/serving-lifecycle/config.json'


def tasks():
    return json.loads(CONFIG.read_text(encoding='utf-8'))['tasks']


def font_preflight(font):
    from PIL import features
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if (importlib.metadata.version('Pillow') != '11.3.0' or not features.check_feature('raqm') or
            hashlib.sha256(Path(font).read_bytes()).hexdigest() != config['font_sha256']):
        raise ValueError('Pinned Pillow, font bytes and RAQM required before ML work')


def verify_git(expected):
    from evaluation.development_runtime import check_digest
    check_digest(expected,40)
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,
                          check=True,timeout=5).stdout.strip()
    if head != expected: raise ValueError('Exact externally reviewed source commit required')
    status = subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,
                            capture_output=True,text=True,check=True,timeout=5).stdout
    if status: raise ValueError('Full clean source checkout required')


def preflight(provider,font,freeze,source_commit):
    from hala.serving.lifecycle import admission
    result = admission(provider)
    verify_sources(freeze)
    verify_git(source_commit)
    font_preflight(font)
    return result


def restart_worker(engine,font):
    # _spawn revalidates source/admission as well; font gate precedes every start.
    font_preflight(font)
    engine.start()


class Workflow:
    """Only this bounded synthetic state machine interprets click proposals."""
    def __init__(self, task, path):
        self.task, self.path = task, Path(path)
        if self.path.exists(): raise ValueError('Fresh outcome path required')
        self.stage, self.opened, self.draft = 0, False, False

    def target_center(self):
        return [200,180]

    def apply(self, point):
        if (self.stage >= 3 or type(point) is not list or len(point)!=2 or
                any(type(v) is not int for v in point) or
                not (120<=point[0]<280 and 140<=point[1]<220)):
            raise ValueError('Invalid synthetic click')
        if self.stage == 0: self.opened = True
        elif self.stage == 1:
            if not self.opened: raise ValueError('Settings not open')
            self.draft = True
        else:
            if not self.opened or not self.draft: raise ValueError('No enabled draft')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x', encoding='utf-8') as output:
                json.dump({'task_id':self.task['id'], 'settings':{self.task['key']:self.draft}}, output)
        self.stage += 1

    def payload(self, image):
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        png = stream.getvalue()
        task = self.task
        rid, frame = task['id']+'-r'+str(self.stage), task['id']+'-frame'+str(self.stage)
        policy = {'schema':'hala.request.v1', 'request_id':rid, 'tenant_id':'h23',
            'task':{'task_id':task['id'], 'goal':task['goal'], 'step_index':self.stage,'max_steps':3},
            'device':{'device_id':'synthetic-h23','session_id':'h23-session','os':'windows-11','locale':task['locale']},
            'observation':{'generation':frame,
                'window':{'window_ref':'h23-window','title':'Synthetic preferences','process':'synthetic.exe'},
                'accessibility':[{'element_ref':frame+'-target','role':'Button',
                                  'name':task['labels'][self.stage], 'bounds':[120,140,160,80]},
                                 {'element_ref':frame+'-protected','role':'Button','name':'Discard',
                                  'bounds':[480,140,160,80],'protected':True}],
                'image':{'format':'png','width':800,'height':600,'coordinate_space':'window','scale':1,'redactions':[]},
                'truncation':{'elements_omitted':0}},
            'allowed_actions':['click','abstain'],'method_ceiling':'WINDOW_TARGETED_INPUT','cancelled':False}
        media = {'png_base64':base64.b64encode(png).decode(), 'sha256':hashlib.sha256(png).hexdigest(),
                 'request_id':rid,'tenant_id':'h23','task_id':task['id'],'device_id':'synthetic-h23',
                 'session_id':'h23-session','frame_id':frame,'origin':[0,0],'dpi':96}
        return {'schema':'hala.image_request.v2','policy_request':policy,'media':media}


def render(workflow, font):
    from PIL import Image, ImageDraw, ImageFont, features
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if hashlib.sha256(Path(font).read_bytes()).hexdigest() != config['font_sha256'] or not features.check_feature('raqm'):
        raise ValueError('Pinned font bytes and RAQM required')
    image = Image.new('RGB',(800,600),'#edf1f7')
    draw = ImageDraw.Draw(image)
    face = ImageFont.truetype(str(font),14)
    rtl = workflow.task['locale'].startswith('ar')
    draw.text((40,40), 'Settings | opened='+str(workflow.opened)+' | draft='+str(workflow.draft), font=face, fill='black')
    draw.rectangle((120,140,279,219), fill='white', outline='black', width=2)
    label = workflow.task['labels'][workflow.stage]
    if rtl:
        draw.text((270,165),label,font=face,fill='black',direction='rtl',anchor='ra')
    else:
        draw.text((125,165),label,font=face,fill='black')
    draw.rectangle((480,140,639,219), fill='#cccccc', outline='black',width=2)
    draw.text((500,165),'Discard',font=face,fill='black')
    return image


def post(port, token, payload):
    connection = http.client.HTTPConnection('127.0.0.1',port,timeout=62)
    try:
        connection.request('POST','/v2/hala/action',body=encode(payload),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        response = connection.getresponse()
        raw = response.read(65537)
        if len(raw)>65536: raise ValueError('Response bound')
        value = strict_json(raw)
        if response.status != 200: raise ServiceError(value['error']['code'])
        return value['result']
    finally:
        connection.close()


def run_tasks(port, token, output, font=None, renderer=render, progress=None):
    records = []
    for task in tasks():
        workflow = Workflow(task,Path(output)/(task['id']+'.json'))
        calls = []
        error = None
        record = {'task_id':task['id'],'calls':calls,'error':None,'verified':False}
        records.append(record)
        if progress: progress(records)
        for _ in range(3):
            payload = workflow.payload(renderer(workflow,font))
            record['pending_request'] = {'request_id':payload['media']['request_id'],
                                         'media_sha256':payload['media']['sha256']}
            if progress: progress(records)
            try:
                proposal = post(port,token,payload)
                apply_proposal(workflow,payload,proposal)
            except (ServiceError, ValueError) as caught:
                error = caught.code if isinstance(caught,ServiceError) else 'invalid_proposal'
                break
            calls.append({'request_id':payload['media']['request_id'],'media_sha256':payload['media']['sha256'],'proposal':proposal})
            record.pop('pending_request',None)
            if progress: progress(records)
        record.update(error=error,verified=verify_outcome(workflow.path,task))
        if progress: progress(records)
    return records


def apply_proposal(workflow,payload,proposal):
    from hala.contracts import Request
    from hala.contracts.proposal import Action, validate_action
    # Explicit independent client binding/policy check before synthetic execution.
    expected = payload['media']
    if (proposal.get('schema') != 'hala.click_proposal.v2' or
            proposal.get('execution_authorized') is not False or
            proposal.get('verification_required') is not True or
            proposal.get('media_sha256') != expected['sha256'] or
            any(proposal.get(key)!=expected[key] for key in ('request_id','tenant_id','task_id','device_id','session_id','frame_id')) or
            len(proposal.get('actions',[])) != 1 or proposal['actions'][0].get('type') != 'click'):
        raise ValueError('Untrusted proposal failed client binding')
    action = Action.from_dict(proposal['actions'][0])
    validate_action(action,Request.from_dict(payload['policy_request']))
    workflow.apply(proposal['actions'][0]['point'])


def run_operator(engine, output, font, progress=None):
    """Explicit remote-owned engine supplied by operator; no default backend activation."""
    token = secrets.token_hex(32)
    server = create_server(Service(Config(token,'h23',request_timeout_s=60,max_concurrency=1),
                                   candidate_backend=CandidateBackend(engine)),port=0)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        return run_tasks(server.server_address[1],token,output,font,progress=progress)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        if not server.wait_for_idle(5): raise RuntimeError('Inference cleanup incomplete')


def probe_lifecycle(engine, font, progress=None):
    """Private destructive tests touch only this operator's child and synthetic task."""
    from hala.inference import RequestContext
    results = {'resource':{'state':'starting'}}
    if progress: progress(results)
    try:
        results['resource'] = engine.allocator_probe(RequestContext(time.monotonic()+10))
    except Exception as error:
        results['resource'].update(error_type=type(error).__name__,fault=engine.last_fault,
                                   wire_trace=list(engine.wire_trace))
        if progress: progress(results)
        raise
    if progress: progress(results)
    for mode in ('disconnect','deadline','worker_failure'):
        results[mode] = {'state':'starting'}
        if progress: progress(results)
        def capture(record):
            results[mode] = record
            if progress: progress(results)
        try:
            results[mode] = probe_fault(engine,font,mode,capture)
        except Exception as error:
            results[mode]['error_type'] = type(error).__name__
            if progress: progress(results)
            raise
        results['reload'] = {'after':mode,'state':'starting'}
        if progress: progress(results)
        try:
            restart_worker(engine,font)
        except Exception as error:
            results['reload']['error_type'] = type(error).__name__
            if progress: progress(results)
            raise
        results['reload']['state'] = 'complete'
        if progress: progress(results)
    return results


def probe_fault(engine,font,mode,progress=None):
    token = secrets.token_hex(32)
    server = create_server(Service(Config(token,'h23',request_timeout_s=.05 if mode=='deadline' else 60,
                                         max_concurrency=1),candidate_backend=CandidateBackend(engine)),port=0)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    sock, trigger = None, None
    record = {'actual_cuda_cancel_qualified':False,'injected_operator_fault':mode=='worker_failure'}
    try:
        # No workflow output is applied during fault probes.
        workflow = Workflow(tasks()[0],Path('/tmp')/('hala-h23-unused-'+secrets.token_hex(8)))
        payload = workflow.payload(render(workflow,font))
        body = encode(payload)
        sock = socket.create_connection(server.server_address,timeout=65)
        engine.wire_trace = []
        engine.last_fault = None
        sock.sendall(('POST /v2/hala/action HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer '+token+
                      '\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body))+'\r\n\r\n').encode()+body)
        if mode != 'deadline':
            until = time.monotonic()+55
            while time.monotonic()<until:
                if any(e.get('event')=='cuda_model_operation_pending' for e in engine.wire_trace):
                    trigger = time.monotonic()
                    break
                if engine.process is None: break
                time.sleep(.001)
            if trigger is None: raise RuntimeError('Pending actual model CUDA operation not observed')
        if mode == 'disconnect':
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        elif mode == 'worker_failure':
            engine.process.terminate()  # exact owned handle; no external PID or command
        response = b''
        if mode != 'disconnect':
            while len(response)<=65536:
                part = sock.recv(min(4096,65537-len(response)))
                if not part: break
                response += part
        record['http_response'] = response.decode('utf-8',errors='replace')
        if not server.wait_for_idle(5): raise RuntimeError('Fault cleanup did not become idle')
        return record
    finally:
        record.update(trigger_monotonic=trigger,fault=engine.last_fault,
                      wire_trace=list(engine.wire_trace),child_reaped=engine.reap_state['confirmed'],
                      reap=dict(engine.reap_state))
        try:
            if progress: progress(record)
        finally:
            if sock is not None: sock.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
            try:
                engine.close()
            finally:
                record['child_reaped'] = engine.reap_state['confirmed']
                record['reap'] = dict(engine.reap_state)
                if progress: progress(record)


def source_files():
    # Freeze the full first-party Python superset, including package initializers,
    # rather than maintaining a fragile hand-written transitive import list.
    names = {p.relative_to(ROOT).as_posix() for folder in ('hala','evaluation','training','data')
             for p in (ROOT/folder).rglob('*.py')}
    names.update(('experiments/serving-lifecycle/config.json','tests/test_serving_lifecycle.py','tests/lifecycle_child.py'))
    return sorted(names)


def verify_sources(expected):
    from evaluation.development_runtime import check_digest
    check_digest(expected)
    freeze = ROOT/'experiments/serving-lifecycle/source-freeze.json'
    if hashlib.sha256(freeze.read_bytes()).hexdigest()!=expected:
        raise ValueError('External prospective source freeze mismatch')
    files = json.loads(freeze.read_text())['files']
    if sorted(files) != source_files(): raise ValueError('Incomplete first-party source freeze')
    for name,digest in files.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen source changed: '+name)


def main():
    import argparse
    from hala.serving.lifecycle import WorkerEngine, PROVENANCE
    from evaluation.development_runtime import persist
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['colab','kaggle'],required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--font',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--faults',action='store_true')
    args = parser.parse_args()
    provider = preflight(args.provider,args.font,args.freeze_sha256,args.source_commit)
    output = Path(args.output)
    if output.exists(): raise ValueError('Fresh experiment directory required')
    engine = WorkerEngine(args.provider,args.font,args.freeze_sha256,args.source_commit)
    output.mkdir(parents=True)
    result = {'schema':'hala.h23.result.v1','qualified':False,'provider':provider,
              'source_freeze_sha256':args.freeze_sha256,'provenance':PROVENANCE,
              'source_commit':args.source_commit,
              'actual_cuda_cancel_qualified':False,'native_qualified':False}
    def save(key,value):
        result[key] = value
        persist(output/'result.json',result)
    try:
        save('load',{'state':'starting'})
        restart_worker(engine,args.font)
        save('load',{'state':'complete'})
        result['tasks'] = run_operator(engine,output/'initial',args.font,lambda value:save('tasks',value))
        if args.faults:
            result['lifecycle'] = probe_lifecycle(engine,args.font,lambda value:save('lifecycle',value))
            result['post_restart_tasks'] = run_operator(engine,output/'post-restart',args.font,lambda value:save('post_restart_tasks',value))
    except Exception as error:
        result['error_type'] = type(error).__name__
        raise
    finally:
        try:
            engine.close()
        finally:
            result['reap'] = dict(engine.reap_state)
            persist(output/'result.json',result)


if __name__=='__main__':
    main()
