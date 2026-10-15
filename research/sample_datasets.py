"""Bounded evaluation-schema inspection only; these samples are not training data."""
from pathlib import Path
import io, json, os, urllib.parse
from PIL import Image
from collect_metadata import fetch, ROOT

def main():
    repo = 'Hcompany/DragOn'
    dest = ROOT / 'Hcompany--DragOn'
    sha = json.loads((dest / 'hub.json').read_text())['sha']
    samples = []
    for domain in ['text_highlight', 'sheet', 'slide_resize', 'slider']:
        path = f'eval/{domain}/tasks'
        tree_url = f'https://huggingface.co/api/datasets/{repo}/tree/{sha}/{path}?limit=1'
        try:
            tree = json.loads(fetch(tree_url))
            entry = next(x for x in tree if x['path'].endswith('.json'))
            task_url = f'https://huggingface.co/datasets/{repo}/resolve/{sha}/' + entry['path']
            tasks = json.loads(fetch(task_url))
            (dest / f'{domain}-sample-tasks.json').write_text(json.dumps(tasks, indent=2), encoding='utf-8')
            img_path = entry['path'].replace('/tasks/', '/images/').removesuffix('.json') + '.jpg'
            image_url = f'https://huggingface.co/datasets/{repo}/resolve/{sha}/{img_path}'
            image_bytes = fetch(image_url)
            image = Image.open(io.BytesIO(image_bytes))
            samples.append({'domain': domain, 'task_url': task_url, 'task_count': len(tasks), 'sample': tasks[0], 'image_url': image_url, 'image_bytes': len(image_bytes), 'image_width': image.width, 'image_height': image.height, 'image_format': image.format})
        except Exception as err:
            samples.append({'domain': domain, 'error': str(err), 'tree_url': tree_url})
    (dest / 'samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in x.items() if k not in ['sample','task_url','image_url']} for x in samples], indent=2))
    profile = Path(os.environ['USERPROFILE'])
    paths = {'kaggle_legacy': profile / '.kaggle/kaggle.json', 'kaggle_access_token': profile / '.kaggle/access_token', 'modal_config': profile / '.modal.toml', 'gcloud_config': profile / 'AppData/Roaming/gcloud', 'hf_token': profile / '.cache/huggingface/token'}
    probe = {'credential_paths_exist': {name:path.exists() for name,path in paths.items()}, 'environment_variable_present': {name: bool(os.environ.get(name)) for name in ['KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY','HF_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET']}, 'account_quota': None, 'billing_state': None, 'note':'Existence only; no credential values were read or printed. Presence does not prove account validity, GPU access or zero spend.'}
    (ROOT / 'credential-existence.json').write_text(json.dumps(probe, indent=2), encoding='utf-8')
    print(json.dumps(probe, indent=2))

if __name__ == '__main__':
    main()
