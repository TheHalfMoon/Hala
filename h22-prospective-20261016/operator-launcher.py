"""Run the externally published fixed six-profile study; never trains or changes a task."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

root=Path('/kaggle/working/hala-h21')
source=root/'source'
commit='07591f109b6683c9302fcf890259ddf0bde91494'
def check_clean_source():
    """Bind the complete first-party Git tree, including transitive imports."""
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    status=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=source,
                          capture_output=True,text=True,check=True)
    if actual!=commit or status.stdout.strip():
        raise ValueError('Exact complete clean published source required before every profile')

check_clean_source()
manifest_sha='3742fc46ec9f4e5151e24f20a78450ab315c25aaae70b002408a64af44a87f39'
manifest_url='https://raw.githubusercontent.com/TheHalfMoon/Hala/253dccf22f54ca0aead4afc3137da1a2276276a0/h22-prospective-20261016/experiment.json'
with urllib.request.urlopen(manifest_url,timeout=45) as response:
    raw=response.read(65537)
assert len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==manifest_sha
manifest=json.loads(raw)
assert manifest['source_commit']==commit
assert manifest['mode_order']==[['fp16','dev'],['int8','dev'],['nf4','dev'],['fp16','test'],['int8','test'],['nf4','test']]
check_clean_source()
qualified=json.loads((root/'h22-qualification.json').read_text())
assert qualified['source_commit']==commit and qualified['gpu_uuid']==manifest['gpu_uuid']
assert qualified['source_freeze_sha256']==manifest['source_freeze_sha256']
results=root/'precision-comparison'
results.mkdir(exist_ok=False)
(results/'experiment.json').write_bytes(raw)
env=dict(os.environ,HALA_REMOTE_DEVELOPMENT_ALLOWED='1',PYTHONIOENCODING='utf-8',
         HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for key in ('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME'):
    env[key]='/tmp/hala-h22-cache'
report={'schema':'hala.h22.actual_launcher.v1','experiment_sha256':manifest_sha,
        'source_commit':commit,'status':'running','runs':[],'training_runs':0,'adoption_qualified':False}
def save():
    path=results/'launcher-status.json'
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(path)
save()
for mode,split in manifest['mode_order']:
    check_clean_source()
    name=mode+'-'+split
    output=results/name
    args=[sys.executable,'-u','-m','evaluation.development_precision',
          '--mode',mode,'--source-freeze-sha256',manifest['source_freeze_sha256'],
          '--bundle',str(root/'frozen'),'--manifest-sha256',manifest['h19_manifest_sha256'],
          '--output',str(output),'--font-path','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
          '--source-commit',commit,'--provider','kaggle','--split',split,
          '--adapter',str(root/'retained-adapter'),'--adapter-manifest-sha256',manifest['adapter_manifest_sha256'],
          '--adapter-identity',json.dumps(manifest['adapter_identity'],sort_keys=True)]
    record={'name':name,'mode':mode,'split':split,'status':'running','argv':args,'started_unix':time.time()}
    report['runs'].append(record)
    save()
    log=results/(name+'.log')
    with log.open('wb') as stream:
        try:
            process=subprocess.run(args,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                   timeout=manifest['budget']['launcher_timeout_seconds_per_run'])
            record.update(returncode=process.returncode,status='finished' if process.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:
            record.update(returncode=None,status='launcher_timeout')
    record['finished_unix']=time.time()
    path=output/'result.json'
    if path.is_file():
        raw_result=path.read_bytes()
        result=json.loads(raw_result)
        record.update(result_sha256=hashlib.sha256(raw_result).hexdigest(),result_status=result['status'],
                      summary=result.get('summary'),adapter_loaded=result['adapter_loaded'],gpu_uuid=result.get('gpu_uuid'))
        if result.get('gpu_uuid')!=manifest['gpu_uuid']:
            record['status']='physical_device_identity_mismatch'
    save()
    archive=results/(name+'-evidence.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')) if output.exists() else []:
            if path.is_file():
                assert path.stat().st_size<=8*1024**2
                z.write(path,str(path.relative_to(results)))
        z.write(log,log.name)
        z.write(results/'experiment.json','experiment.json')
        for filename in ('h22-qualification.json','h22-preparation.json'):
            z.write(root/filename,filename)
        z.write(Path(__file__),'operator-launcher.py')
        z.write(source/'experiments/development-precision/source-freeze.json','source-freeze.json')
    assert archive.stat().st_size<=16*1024**2
    record.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    save()
    print('HALA_H22_RUN_COMPLETE:'+json.dumps(record,sort_keys=True),flush=True)
report['status']='finished' if all(r['status']=='finished' and r.get('result_status')=='measured' for r in report['runs']) else 'finished_with_blocked_profiles'
save()
print('HALA_H22_STUDY_COMPLETE:'+json.dumps(report,sort_keys=True),flush=True)
