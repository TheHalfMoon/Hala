from pathlib import Path
import json
from collect_metadata import ROOT, IDS, fetch

for repo in IDS:
    dest = ROOT / repo.replace('/', '--')
    hub = json.loads((dest / 'hub.json').read_text())
    config = json.loads((dest / 'config.json').read_text())
    url = f'https://huggingface.co/api/models/{repo}/tree/{hub["sha"]}'
    tree = json.loads(fetch(url))
    (dest / 'tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
    weights = [x for x in tree if x.get('path','').endswith('.safetensors')]
    proc = {}
    for name in ['preprocessor_config.json', 'processor_config.json']:
        if (dest / name).exists():
            proc[name] = json.loads((dest / name).read_text())
    print(json.dumps({'id':repo,'sha':hub['sha'],'params':hub.get('safetensors'), 'architectures':config.get('architectures'),'transformers':config.get('transformers_version'),'dtype':config.get('dtype',config.get('torch_dtype')),'disk_bytes':sum(x['size'] for x in weights),'processors':proc}))

checks = {'transformers-4.57.6':'https://pypi.org/pypi/transformers/4.57.6/json', 'apache-2.0':'https://www.apache.org/licenses/LICENSE-2.0.txt', 'qwen3-vl-source':'https://api.github.com/repos/QwenLM/Qwen3-VL/commits/main'}
for name,url in checks.items():
    try:
        data=fetch(url)
        (ROOT / (name+'.txt')).write_bytes(data)
        print(name,'available')
        if name=='qwen3-vl-source':
            sha=json.loads(data)['sha']
            (ROOT / 'qwen3-vl-source-LICENSE').write_bytes(fetch(f'https://raw.githubusercontent.com/QwenLM/Qwen3-VL/{sha}/LICENSE'))
            print('qwen3-vl-source revision',sha)
    except Exception as e:
        print(name,str(e))
