"""Summarize retained actual DEV records; no model execution or adoption."""
from pathlib import Path
import hashlib
import json
import sys

w=Path(__file__).parent.parent
sys.path.insert(0,str(w/'work/Hala-model'))
from evaluation.development_precision import paired
rows=[];results=[]
for index,mode in enumerate(('fp16','int8','nf4')):
    path=w/f'outputs/h22-attempt2-profile-{index}-{mode}-dev-evidence/run/result.json'
    raw=path.read_bytes();d=json.loads(raw);results.append(d)
    assert d['status']=='measured' and d['gpu_uuid']=='GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1'
    rows.append({'mode':mode,'result_sha256':hashlib.sha256(raw).hexdigest(),
        'summary':d['summary'], 'wall_seconds':d['wall_seconds'],'load_seconds':d['load_seconds'],
        'model_generation_seconds':d['model_generation_seconds'],
        'memory_final':d['memory_final'],'residency_loaded':{k:v for k,v in d['residency_loaded'].items() if k!='tensors'},
        'unique_physical_cache_bytes':d['unique_physical_cache_bytes'],
        'calls':len(d['calls']),'quality_qualified':False,'adoption_qualified':False})
comparison={'schema':'hala.h22.actual_dev_comparison.v1','split':'dev',
 'source_commit':'07591f109b6683c9302fcf890259ddf0bde91494','rows':rows,
 'runtime_secondary_pairs':{mode:paired(results[0],results[i]) for i,mode in ((1,'int8'),(2,'nf4'))},
 'independent_audit_checks':[9114,10972,12537],
 'test_profiles_run':0,'test_quality_or_cost_conclusion':False,
 'power_qualified':False,'adoption_qualified':False,'native_qualified':False,
 'limits':['DEV only; remaining TEST profiles did not run before session expiry',
           'Adaptive trajectories have different prompts/reached states; descriptive aggregate comparison',
           'Same recorded physical GPU; fixed temporal order and warm cache confound timing',
           'Power readings exceed a later observed limit and remain uncalibrated',
           'Raw data all retained before VM expiry; no rerun or forgiving rescore']}
for paired_record in comparison['runtime_secondary_pairs'].values():
    for cost in paired_record['cost'].values():
        if cost.get('residency_loaded'):
            cost['residency_loaded']={k:v for k,v in cost['residency_loaded'].items() if k!='tensors'}
(w/'outputs/h22-dev-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
for row in rows:
    print(json.dumps({k:row[k] for k in ('mode','wall_seconds','load_seconds','calls')}|{
      'summary':row['summary']['all'],'memory_final':row['memory_final'],
      'storage':row['residency_loaded'],'cache':row['unique_physical_cache_bytes']}))
