"""Read-only, bounded official Hub metadata capture. Never fetch weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, urllib.request, urllib.error

ROOT = Path(__file__).parent / 'evidence'
ROOT.mkdir(exist_ok=True)
IDS = ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3-VL-2B-Instruct', 'Hcompany/Holo-3.1-4B', 'Hcompany/Holo-3.1-0.8B', 'showlab/ShowUI-2B', 'Qwen/Qwen3-VL-4B-Instruct', 'Hcompany/Holo4-35B-A3B', 'LiquidAI/d1-omni-600M', 'LiquidAI/d1-3B']
ALLOWED = {'README.md', 'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'config.json', 'preprocessor_config.json', 'processor_config.json', 'model.safetensors.index.json'}

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Hala-source-research/0.1'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError('metadata exceeds 2 MB budget')
    return data

def capture(repo, kind='model'):
    dest = ROOT / repo.replace('/', '--')
    dest.mkdir(exist_ok=True)
    api = 'https://huggingface.co/api/' + ('models/' if kind == 'model' else 'datasets/') + repo
    if kind == 'dataset':
        api += '?expand[]=sha&expand[]=cardData&expand[]=description'
    info = json.loads(fetch(api))
    (dest / 'hub.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    sources = [{'url': api, 'type': 'hub_api', 'revision': info['sha']}]
    entries = info.get('siblings', []) if kind == 'model' else [{'rfilename': 'README.md'}]
    for entry in entries:
        name = entry['rfilename']
        if name not in ALLOWED:
            continue
        url = 'https://huggingface.co/' + ('datasets/' if kind == 'dataset' else '') + repo + '/resolve/' + info['sha'] + '/' + name
        try:
            data = fetch(url)
            (dest / name).write_bytes(data)
            sources.append({'url': url, 'file': name, 'sha256': hashlib.sha256(data).hexdigest()})
        except Exception as err:
            sources.append({'url': url, 'error': str(err)})
    (dest / 'provenance.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print(repo, info['sha'], info.get('cardData', {}).get('license'), flush=True)

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(capture, IDS))
    for repo in ['Hcompany/DragOn', 'Hcompany/trajectories']:
        capture(repo, 'dataset')
        dest = ROOT / repo.replace('/', '--')
        for endpoint in ['splits', 'size']:
            url = 'https://datasets-server.huggingface.co/' + endpoint + '?dataset=' + repo
            try:
                (dest / (endpoint + '.json')).write_bytes(fetch(url))
            except Exception as err:
                (dest / (endpoint + '.error.txt')).write_text(str(err), encoding='utf-8')
