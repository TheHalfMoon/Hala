import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('index',type=int,choices=range(3));parser.add_argument('commit')
args=parser.parse_args()
assert re.fullmatch('[0-9a-f]{40}',args.commit)
workspace=Path(__file__).parent.parent
mode,split=[('fp16','test'),('int8','test'),('nf4','test')][args.index]
name=f'profile-{args.index}-{mode}-{split}'
records=[]
for suffix,limit in [('-evidence.zip',16*1024**2),('-receipt.json',65536)]:
    local=workspace/'work/Hala-artifacts/h22-test-restart'/(name+suffix)
    url=f'https://raw.githubusercontent.com/TheHalfMoon/Hala/{args.commit}/h22-test-restart/{name}{suffix}'
    with urllib.request.urlopen(url,timeout=45) as response:
        assert response.geturl()==url
        raw=response.read(limit+1)
    assert len(raw)<=limit and raw==local.read_bytes()
    records.append({'path':'h22-test-restart/'+name+suffix,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
record={'schema':'hala.h22.public_roundtrip.v1','index':args.index,'commit':args.commit,'verified':True,'files':records}
(workspace/'outputs'/f'h22-test-restart-profile-{args.index}-public-roundtrip.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
