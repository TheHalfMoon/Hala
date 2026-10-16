"""One private H23 run, only after all six H22 profiles are publicly retained."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

WORK = Path('/kaggle/working')
ROOT = WORK / 'hala-h23'
H22 = WORK / 'hala-h21'
EVIDENCE = WORK / 'h23-operator-evidence'
H22_OPERATOR = WORK / 'h22-test-only-operator.py'
H22_OPERATOR_SHA = 'd8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e'
H22_EXPERIMENT = '60a0986cc196041c2474314624e0f88c9c2986d75bc58cf0ddb44623ed294e02'
SOURCE = '18ad45deaa890c27c7250681e4bffe8b881e5690'
FREEZE = '4c52241c1312648ae08d2530f709b45ed3fec41ae8924d2a03df5b431fcefc8a'
PROSPECT_COMMIT = 'd0e2115456be9f199841b6df44531b55467ccca0'
PROSPECT_SHA = '399b094ba74dde4b7c3f6aa2551f9d0d0eaf6df86c236332481e6a5b06c27592'
PREFIX = 'https://raw.githubusercontent.com/TheHalfMoon/Hala/'
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FONT_SHA = 'ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280'
OLD_CACHE, CACHE = Path('/tmp/hala-h22-cache'), Path('/tmp/hala-h23-cache')
CACHE_KEYS = ('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME')
TEST_ORDER = [['fp16','test'],['int8','test'],['nf4','test']]
TERMINAL = ('finished','failed','launcher_timeout','launcher_error','physical_device_identity_mismatch')
MIB = 1024**2
PINS = {'torch':'2.11.0+cu128','transformers':'5.16.1','Pillow':'11.3.0',
        'accelerate':'1.12.0','bitsandbytes':'0.50.2','peft':'0.20.0','safetensors':'0.8.0'}
EXECUTION_READY = True  # All runtime amendment, public retention and hardware guards still mandatory.
AMENDMENT_PATH = 'h23-hardware-retention-amendment/experiment.json'
TABLE_PATH = 'h23-hardware-retention-amendment/retention-table.json'
SEQUENCE = ['original_DEV_public_retention_0_1_2','new_TEST_public_and_local_retention_0_1_2',
            'H23_single_fault_run_and_immediate_export']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value,length=64):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is None:
        raise ValueError('Immutable lowercase digest required')


def read(path,limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('Bounded non-symlink file required')
    with path.open('rb') as stream: raw = stream.read(limit+1)
    if len(raw)>limit: raise ValueError('File changed beyond bound')
    return raw


def fetch(url,expected,limit):
    digest(expected)
    if re.fullmatch(re.escape(PREFIX)+r'[0-9a-f]{40}/[A-Za-z0-9_./-]+',url) is None or '/..' in url:
        raise ValueError('Fixed public immutable raw URL required')
    with urllib.request.urlopen(url,timeout=45) as response:
        if response.geturl()!=url: raise ValueError('Redirect rejected')
        raw = response.read(limit+1)
    if len(raw)>limit or sha(raw)!=expected: raise ValueError('Public bytes/digest mismatch')
    return raw


def atomic(path,value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    if len(raw)>8*MIB: raise ValueError('Evidence JSON bound')
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def genuine_provider():
    if (platform.system()!='Linux' or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') not in ('Interactive','Batch') or
            not WORK.is_dir() or WORK.is_symlink()):
        raise ValueError('Actual admitted Linux Kaggle runtime required; no local execution')


def verified_h22_operator(path=H22_OPERATOR):
    raw = read(path,65536)
    if sha(raw)!=H22_OPERATOR_SHA:
        raise ValueError('Reviewed TEST retention helper bytes changed before import')
    spec = importlib.util.spec_from_file_location('h23_verified_h22_retention',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_amendment(amendment,test_manifest,helper,original):
    fixed = {'schema':'hala.h23.hardware_retention_amendment.v1',
             'original_prospective_commit':PROSPECT_COMMIT,'original_prospective_sha256':PROSPECT_SHA,
             'source_commit':SOURCE,'source_freeze_sha256':FREEZE,'dev_experiment_sha256':H22_EXPERIMENT,
             'execution_sequence':SEQUENCE,'cross_gpu_cost_pairing':False,'scientific_changes':False,
             'actual_gpu_execution_started':False,'actual_qualification':False}
    if set(amendment)!=set(fixed)|{'test_manifest_commit','test_manifest_sha256','h23_expected_gpu_uuid'}:
        raise ValueError('Exact engineering-only H23 hardware amendment required')
    if any(json.dumps(amendment[key],sort_keys=True,allow_nan=False)!=json.dumps(value,sort_keys=True,allow_nan=False)
           for key,value in fixed.items()):
        raise ValueError('Original H23 source/science or sequence changed')
    digest(amendment['test_manifest_commit'],40)
    digest(amendment['test_manifest_sha256'])
    helper.validate_manifest(test_manifest,original)
    if (test_manifest.get('schema')!='hala.h22.test_restart_amendment.v1' or test_manifest.get('attempt')!=3 or
            test_manifest.get('mode_order')!=TEST_ORDER or test_manifest.get('dev_experiment_sha256')!=H22_EXPERIMENT or
            test_manifest.get('dev_retention')!=helper.DEV_RETENTION or
            amendment['h23_expected_gpu_uuid']!=test_manifest.get('gpu_uuid') or
            re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',str(amendment['h23_expected_gpu_uuid'])) is None):
        raise ValueError('Separate frozen TEST group and same subsequent H23 hardware required')


def barrier(table_raw,table_sha,helper,test_manifest,test_sha,root=H22):
    digest(table_sha)
    if len(table_raw)>65536 or sha(table_raw)!=table_sha: raise ValueError('External retention table SHA mismatch')
    table = json.loads(table_raw)
    if (set(table)!={'schema','dev_experiment_sha256','test_experiment_sha256','profiles'} or
            table['schema']!='hala.h23.two_group_retention.v1' or
            table['dev_experiment_sha256']!=H22_EXPERIMENT or table['test_experiment_sha256']!=test_sha or
            type(table['profiles']) is not list or len(table['profiles'])!=6):
        raise ValueError('Exactly two separate frozen groups and six retained rows required')
    checked = []
    for slot,row in enumerate(table['profiles']):
        group, index = ('dev',slot) if slot<3 else ('test',slot-3)
        if (type(row) is not dict or set(row)!={'group','index','commit','receipt_sha256','archive_sha256'} or
                row['group']!=group or
                type(row['index']) is not int or row['index']!=index):
            raise ValueError('Ordered unique fixed profile indexes required')
        digest(row['commit'],40)
        digest(row['receipt_sha256'])
        digest(row['archive_sha256'])
        if group=='dev' and {k:v for k,v in row.items() if k!='group'}!=helper.DEV_RETENTION[index]:
            raise ValueError('Original immutable DEV retention rows changed')
        experiment_sha = H22_EXPERIMENT if group=='dev' else test_sha
        name = 'profile-'+str(index)+'-'+('fp16','int8','nf4')[index]+'-'+group
        prefix = 'h22-attempt2/' if group=='dev' else 'h22-test-restart/'
        # Reviewed helper fetches and verifies both exact public receipt and raw ZIP,
        # fixed mode/split/source/result identity, CRC and ZIP size/member bounds.
        receipt = helper.prior_retention(index+1,experiment_sha,row['commit'],row['receipt_sha256'],row['archive_sha256'],group=group)
        if group=='test':
            status = json.loads(read(root/'precision-test-restart'/name/'profile-status.json',65536))
            archive_raw = read(root/'precision-test-restart'/(name+'-evidence.zip'),16*MIB)
            if (status.get('schema')!='hala.h22.single_profile.v1' or status.get('status') not in TERMINAL or
                    type(status.get('index')) is not int or status['index']!=index or status.get('name')!=name or
                    status.get('source_commit')!=helper.COMMIT or status.get('experiment_sha256')!=test_sha or
                    status.get('archive_sha256')!=row['archive_sha256'] or sha(archive_raw)!=row['archive_sha256']):
                raise ValueError('New TEST local terminal status/raw ZIP barrier failed')
            with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
                archived = json.loads(archive.read('profile-status.json'))
                qualification = json.loads(archive.read('qualification.json'))
                manifest_raw = archive.read('experiment.json')
            if (any(status.get(key)!=value for key,value in archived.items()) or sha(manifest_raw)!=test_sha or
                    qualification.get('gpu_uuid')!=test_manifest['gpu_uuid']):
                raise ValueError('TEST local final/raw status or frozen hardware differs')
        checked.append({'group':group,'index':index,'experiment_sha256':experiment_sha,
                        'commit':row['commit'],'receipt_sha256':row['receipt_sha256'],
                        'archive_sha256':row['archive_sha256'],'receipt':receipt,
                        'archive_url':PREFIX+row['commit']+'/'+prefix+name+'-evidence.zip',
                        'receipt_url':PREFIX+row['commit']+'/'+prefix+name+'-receipt.json',
                        'public_raw_verified':True,'local_raw_verified':group=='test'})
    return checked


def clean_source(source):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True,timeout=30).strip()
    dirty = subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=source,
                           check=True,capture_output=True,text=True,timeout=30).stdout
    if head!=SOURCE or dirty: raise ValueError('Exact clean reviewed H23 source required')


def prepare_source_and_cache():
    if os.path.lexists(ROOT): raise FileExistsError('One fresh H23 root required')
    ROOT.mkdir()
    source = ROOT/'source'
    subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/TheHalfMoon/Hala.git',str(source)],check=True,timeout=300)
    subprocess.run(['git','checkout','--detach',SOURCE],cwd=source,check=True,timeout=120)
    clean_source(source)
    sys.path.insert(0,str(source))
    from evaluation.serving_lifecycle import verify_sources
    from evaluation.development_precision import disk_bytes
    verify_sources(FREEZE)
    if (OLD_CACHE.is_symlink() or OLD_CACHE.resolve()!=OLD_CACHE or not OLD_CACHE.is_dir() or
            os.path.lexists(CACHE)):
        raise ValueError('Exact owned old cache and nonexistent new cache required')
    disk_bytes(OLD_CACHE)  # Reviewed bounded symlink/entry/time scan before rename.
    OLD_CACHE.rename(CACHE)
    for key in CACHE_KEYS: os.environ[key] = str(CACHE)
    os.environ.update(HALA_REMOTE_DEVELOPMENT_ALLOWED='1',PYTHONIOENCODING='utf-8',
                      HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
    return source


def qualify(source,expected_gpu):
    clean_source(source)
    from evaluation.serving_lifecycle import preflight
    provider = preflight('kaggle',str(FONT),FREEZE,SOURCE)
    versions = {key:importlib.metadata.version(key) for key in PINS}
    if versions!=PINS or importlib.util.find_spec('torchao') is not None:
        raise ValueError('Frozen environment pins/absent TorchAO required')
    if sha(read(FONT,8*MIB))!=FONT_SHA: raise ValueError('Pinned font changed')
    import torch
    import transformers
    from PIL import features
    from evaluation.development_precision import resolve_device_uuid
    if (torch.__version__!=PINS['torch'] or transformers.__version__!=PINS['transformers'] or
            not torch.cuda.is_available() or features.version_feature('raqm')!='0.10.1' or
            resolve_device_uuid(torch)!=expected_gpu):
        raise ValueError('Actual pinned imports, RAQM and same physical GPU0 required')
    physical = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip()
    if physical!=expected_gpu: raise ValueError('Physical GPU0 changed')
    timeout_version = subprocess.check_output(['/usr/bin/timeout','--version'],text=True,timeout=5)
    if not timeout_version.startswith('timeout (GNU coreutils)'): raise ValueError('Official coreutils timeout required')
    return {'provider':provider,'packages':versions,'gpu_uuid':expected_gpu,'physical_gpu0_uuid':physical,
            'raqm':'0.10.1','font_sha256':FONT_SHA,'model_loaded':False,
            'source_commit':SOURCE,'source_freeze_sha256':FREEZE,'timeout_version':timeout_version.splitlines()[0]}


def finish_owned(process,abort):
    """Stop/reap only the process group created by our start_new_session Popen."""
    state={'owned_group':process.pid,'leader_reaped':False,'group_quiescent':False,
           'confirmed':False,'returncode':None,'signals':[],'errors':[]}
    def send(sig):
        try:
            os.killpg(process.pid,sig)
            state['signals'].append(int(sig))
        except ProcessLookupError: pass
        except Exception as error: state['errors'].append('signal:'+type(error).__name__)
    def gone():
        try: os.killpg(process.pid,0)
        except ProcessLookupError: return True
        except Exception as error: state['errors'].append('probe:'+type(error).__name__)
        return False
    def wait(seconds):
        try:
            code=process.wait(timeout=seconds)
            if type(code) is not int: raise ValueError('Exit not confirmed')
            state.update(leader_reaped=True,returncode=code)
        except Exception as error: state['errors'].append('wait:'+type(error).__name__)
    if abort: send(signal.SIGTERM)
    wait(5 if abort else 1)
    if not state['leader_reaped']:
        send(signal.SIGKILL)
        wait(2)
    state['group_quiescent']=gone()
    if not state['group_quiescent']:
        if int(signal.SIGTERM) not in state['signals']: send(signal.SIGTERM)
        until=time.monotonic()+.5
        while time.monotonic()<until and not gone(): time.sleep(.01)
        if not gone():
            send(signal.SIGKILL)
            until=time.monotonic()+.5
            while time.monotonic()<until and not gone(): time.sleep(.01)
        state['group_quiescent']=gone()
    state['confirmed']=state['leader_reaped'] and state['group_quiescent']
    return state


def invoke(source,evidence,report):
    args = ['/usr/bin/timeout','--signal=TERM','--kill-after=5s','1200s',sys.executable,'-u','-m',
            'evaluation.serving_lifecycle','--provider','kaggle','--freeze-sha256',FREEZE,
            '--source-commit',SOURCE,'--font',str(FONT),'--output',str(evidence/'actual-run'),'--faults']
    report.update(argv=args,cwd=str(source),status='running',all_qualification_false=True)
    atomic(evidence/'operator-status.json',report)
    # timeout owns only this new invocation group. No foreground mode, external
    # PID, shell command, tunnel, alternate source/model or automatic rerun.
    total, forced_incomplete, monitor_error = 0, False, None
    process = subprocess.Popen(args,cwd=source,env=dict(os.environ),stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,start_new_session=True)
    try:
        report.update(owned_timeout_pid=process.pid,producer_quiescent=False)
        deadline=time.monotonic()+1208
        with selectors.DefaultSelector() as selector, (evidence/'invocation.log').open('ab') as stream:
            selector.register(process.stdout,selectors.EVENT_READ)
            while True:
                remaining_time=deadline-time.monotonic()
                if remaining_time<=0:
                    forced_incomplete=True
                    break
                if not selector.select(min(.5,remaining_time)): continue
                piece=os.read(process.stdout.fileno(),65536)
                if not piece: break
                remaining=max(0,8*MIB-total)
                stream.write(piece[:remaining])
                total+=len(piece)
    except BaseException as error:
        monitor_error=type(error).__name__
        report['monitor_error_type']=monitor_error
        raise
    finally:
        cleanup=finish_owned(process,abort=forced_incomplete or monitor_error is not None)
        code=cleanup['returncode']
        report.update(cleanup=cleanup,producer_quiescent=cleanup['confirmed'],returncode=code,
                      owned_timeout_reaped=cleanup['leader_reaped'],log_bytes_observed=total,log_truncated=total>8*MIB)
        if not cleanup['confirmed']: report['status']='incomplete_cleanup'
        elif monitor_error: report['status']='failed'
        elif forced_incomplete or code in (124,137): report['status']='incomplete_timeout'
        else: report['status']='finished' if code==0 else 'failed'
        if total>8*MIB and cleanup['confirmed']: report['status']='incomplete_log_bound'
        try: process.stdout.close()
        except Exception as error: report['stdout_close_error_type']=type(error).__name__
        atomic(evidence/'operator-status.json',report)
    return report


def retain(evidence,report):
    atomic(evidence/'operator-status.json',report)
    if report.get('producer_quiescent') is False:
        export={'archive':None,'status':'incomplete_cleanup','actual_qualification':False,
                'retention_blocked':'owned producer not confirmed quiescent','raw_evidence':str(evidence),
                'cleanup':report.get('cleanup')}
        atomic(evidence.with_name(evidence.name+'-export.json'),export)
        print('HALA_H23_RETENTION_BLOCKED:'+json.dumps(export,sort_keys=True),flush=True)
        return export
    files = sorted(p for p in evidence.rglob('*') if p.is_file() or p.is_symlink())
    sizes = [len(read(p,8*MIB)) for p in files]
    if len(files)>1000 or sum(sizes)>64*MIB: raise ValueError('Evidence inventory/total bound')
    target = evidence.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.writestr(path.relative_to(evidence).as_posix(),read(path,8*MIB))
    if target.stat().st_size>16*MIB: raise ValueError('Archive bound; raw evidence remains on disk')
    export = {'archive':str(target),'archive_sha256':sha(read(target,16*MIB)),
              'archive_bytes':target.stat().st_size,'status':report['status'],'actual_qualification':False}
    atomic(evidence.with_name(evidence.name+'-export.json'),export)
    print('HALA_H23_EXPORT_NOW:'+json.dumps(export,sort_keys=True),flush=True)
    return export


def run(amendment_commit,amendment_sha,retention_table_commit,retention_table_sha):
    if not EXECUTION_READY:
        raise ValueError('Blocked: explicit DEV/TEST hardware-retention amendment and final independent review required')
    genuine_provider()
    if os.path.lexists(EVIDENCE) or os.path.lexists(EVIDENCE.with_suffix('.zip')) or os.path.lexists(ROOT):
        raise FileExistsError('One unattended H23 invocation only; preserve earlier evidence')
    EVIDENCE.mkdir()
    report = {'schema':'hala.h23.operator.v1','status':'preflight','source_commit':SOURCE,
              'source_freeze_sha256':FREEZE,'started_unix':time.time(),'actual_qualification':False,
              'actual_cuda_cancel_qualified':False,'native_qualified':False,'production_qualified':False}
    try:
        (EVIDENCE/'invocation.log').touch(exist_ok=False)  # Empty means no source invocation began.
        (EVIDENCE/'operator.py').write_bytes(read(Path(__file__),65536))
        (EVIDENCE/'experiment.json').write_bytes(fetch(PREFIX+PROSPECT_COMMIT+'/h23-prospective-20261016/experiment.json',PROSPECT_SHA,65536))
        (EVIDENCE/'source-freeze.json').write_bytes(fetch(PREFIX+SOURCE+'/experiments/serving-lifecycle/source-freeze.json',FREEZE,65536))
        helper = verified_h22_operator()
        digest(amendment_commit,40)
        digest(retention_table_commit,40)
        amendment_raw = fetch(PREFIX+amendment_commit+'/'+AMENDMENT_PATH,amendment_sha,65536)
        (EVIDENCE/'hardware-retention-amendment.json').write_bytes(amendment_raw)
        amendment = json.loads(amendment_raw)
        digest(amendment.get('test_manifest_commit'),40)
        digest(amendment.get('test_manifest_sha256'))
        test_sha = amendment['test_manifest_sha256']
        test_raw = fetch(PREFIX+amendment['test_manifest_commit']+'/h22-test-restart/experiment.json',test_sha,65536)
        (EVIDENCE/'test-group-experiment.json').write_bytes(test_raw)
        test_manifest = json.loads(test_raw)
        original_raw = fetch(helper.ORIGINAL_URL,helper.ORIGINAL_SHA,65536)
        (EVIDENCE/'original-precision-experiment.json').write_bytes(original_raw)
        validate_amendment(amendment,test_manifest,helper,json.loads(original_raw))
        table_raw = fetch(PREFIX+retention_table_commit+'/'+TABLE_PATH,retention_table_sha,65536)
        (EVIDENCE/'retention-table.json').write_bytes(table_raw)
        checked = barrier(table_raw,retention_table_sha,helper,test_manifest,test_sha)
        atomic(EVIDENCE/'six-profile-barrier.json',{'passed':True,'profiles':checked,'table_sha256':retention_table_sha,
                                                 'dev_experiment_sha256':H22_EXPERIMENT,'test_experiment_sha256':test_sha})
        report['six_profile_barrier_passed'] = True
        atomic(EVIDENCE/'operator-status.json',report)
        source = prepare_source_and_cache()  # Strictly after every local/public check.
        atomic(EVIDENCE/'qualification.json',qualify(source,amendment['h23_expected_gpu_uuid']))
        clean_source(source)
        invoke(source,EVIDENCE,report)
    except Exception as error:
        report.update(status='incomplete_cleanup' if report.get('producer_quiescent') is False else 'failed',
                      error_type=type(error).__name__,error_message=str(error)[:512])
    finally:
        report['finished_unix'] = time.time()
        retain(EVIDENCE,report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment-commit',required=True)
    parser.add_argument('--amendment-sha',required=True)
    parser.add_argument('--retention-table-commit',required=True)
    parser.add_argument('--retention-table-sha',required=True)
    args = parser.parse_args()
    run(args.amendment_commit,args.amendment_sha,args.retention_table_commit,args.retention_table_sha)
