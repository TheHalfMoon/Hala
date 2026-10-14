"""Offline integrity audit of captured evidence; not model qualification."""
from pathlib import Path
import hashlib, json, re

root=Path(__file__).parent
registry=json.loads((root.parent/'hala/models/registry.json').read_text(encoding='utf-8'))
checked=0
for candidate in registry['candidates']:
    assert re.fullmatch('[a-f0-9]{40}',candidate['revision']), candidate['id']
    assert candidate['status']=='unqualified' and not candidate['runtime_compatibility_tested']
    assert all(value is None for value in candidate['measurements'].values())
    directory=root/'evidence'/candidate['id'].replace('/','--')
    metadata=json.loads((directory/'hub.json').read_text(encoding='utf-8'))
    assert metadata['sha']==candidate['revision']
    for source in json.loads((directory/'provenance.json').read_text(encoding='utf-8')):
        if 'file' not in source: continue
        assert candidate['revision'] in source['url']
        data=(directory/source['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==source['sha256'],source['file']
        checked+=1
    assert not list(directory.glob('*.safetensors')) and not list(directory.glob('*.bin'))
    if candidate['id'].startswith('LiquidAI/'):
        license_text=(directory/'LICENSE').read_text(encoding='utf-8')
        assert 'LFM Open License v1.0' in license_text and '$10,000,000' in license_text
        assert candidate['license_status']=='conditional-commercial'
samples=json.loads((root/'evidence/Hcompany--DragOn/samples.json').read_text(encoding='utf-8'))
assert len(samples)==4
for entry in samples:
    assert 'error' not in entry
    assert 'eval/' in entry['task_url'] and '3451c1fc3543c18fc3361a32bfe1f12cc69b64b5' in entry['task_url']
    for field in ['start_bbox','end_bbox']:
        x0,y0,x1,y1=entry['sample'][field]
        assert 0 <= x0 <= x1 <= entry['image_width'] and 0 <= y0 <= y1 <= entry['image_height']
report={'date':'2026-10-14','candidate_count':len(registry['candidates']),'pinned_file_hashes_checked':checked,'drag_eval_samples_checked':len(samples),'weights_downloaded_locally':False,'model_runtime_tested_by_research_builder':False,'integrity_check':'pass','limitations':'Checks captured file integrity/schema and original-dimension sample bounds, not model correctness, dataset rights, provider eligibility or performance.'}
(root/'evidence/integrity-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
