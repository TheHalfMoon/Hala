"""Retain an externally hash-verified small raw H22 profile and receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('index',type=int,choices=range(3))
parser.add_argument('sha256')
args=parser.parse_args()
workspace=Path(__file__).parent.parent
order=[('fp16','test'),('int8','test'),('nf4','test')]
name='profile-'+str(args.index)+'-'+'-'.join(order[args.index])
archive=workspace/'outputs'/('h22-test-restart-'+name+'-evidence.zip')
raw=archive.read_bytes()
assert len(raw)<=16*1024**2 and hashlib.sha256(raw).hexdigest()==args.sha256
dest=workspace/'work/Hala-artifacts/h22-test-restart'
experiment=(dest/'experiment.json').read_bytes()
experiment_sha=hashlib.sha256(experiment).hexdigest()
assert experiment_sha=='21cab7460e729b4f02190d03ba368a468a330477e902d11f694e3a3e8aaf6c99'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert z.read('experiment.json')==experiment
    status=json.loads(z.read('profile-status.json'))
    assert status['index']==args.index and status['name']==name and status['experiment_sha256']==experiment_sha
    assert status['status'] in ('finished','failed','launcher_timeout','launcher_error','physical_device_identity_mismatch')
    for member in ('profile.log','qualification.json','operator-launcher.py'):
        assert member in z.namelist()
    if status.get('result_sha256'):
        assert hashlib.sha256(z.read('run/result.json')).hexdigest()==status['result_sha256']
target=dest/(name+'-evidence.zip')
assert not target.exists()
shutil.copyfile(archive,target)
receipt={'schema':'hala.h22.public_retention_receipt.v1','index':args.index,
         'experiment_sha256':experiment_sha,'archive_sha256':args.sha256,
         'archive_path':'h22-test-restart/'+name+'-evidence.zip'}
receipt_path=dest/(name+'-receipt.json')
assert not receipt_path.exists()
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'name':name,'archive_bytes':len(raw),'archive_sha256':args.sha256,
                  'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                  'status':status['status'],'result_sha256':status.get('result_sha256')}))
