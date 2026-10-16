"""Independent stdlib-only audit. Does not import production implementation."""
import copy, hashlib, json, re, struct, zipfile, subprocess
import sys
SPLIT=sys.argv[1] if len(sys.argv)>1 else 'dev'
EXPECTED={'dev':('0619aa121f044e26fbe1be8533f3ade5c2279c333a0b09bf70ba2448d51bfb25',263482,'ab57e72fc5efc9badb394fe11272e6896713c2481896630e25418e3e8e4b1b15'),'test':('be538578cd0e0f3eb6aef4155d9e6a0625a04169ce32d3fefb1bae4cf6649c1c',295810,'1de1a9f270b5dfa730c04ef258ceb4f26aabc2a95610ebbaef2d8a1f9ae19831')}[SPLIT]
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v): return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
errors=[]; checks=0
def check(ok, name):
    global checks
    checks+=1
    if not ok: errors.append(name)
evidence=ROOT/f'outputs/h21-kaggle-retry1-adapter-{SPLIT}-evidence'
result=evidence/f'adapter-{SPLIT}/result.json'; d=read(result)
freeze=ROOT/'outputs/h21-frozen-data'; dataset=read(freeze/'dataset.json'); manifest=read(freeze/'manifest.json')
archive=ROOT/f'outputs/h21-kaggle-retry1-adapter-{SPLIT}-evidence.zip'
check(sha(archive)==EXPECTED[0],'archive sha')
check(archive.stat().st_size==EXPECTED[1],'archive bytes')
check(sha(result)==EXPECTED[2],'result sha')
with zipfile.ZipFile(archive) as z:
    check(z.testzip() is None,'zip CRC')
    for name in z.namelist():
        if not name.endswith('/'): check((evidence/name).read_bytes()==z.read(name),'archive extraction '+name)
check(sha(freeze/'manifest.json')==d['manifest_sha256'],'freeze manifest')
for name, digest in manifest['files'].items(): check(sha(freeze/name)==digest,'frozen '+name)
for name,digest in d['frozen_source'].items():
    check(sha(ROOT/'work/Hala-model/development'/name)==digest==manifest['source'][name],'source '+name)
runtime_source=subprocess.run(['git','-C',str(ROOT/'work/Hala-model'),'show',d['source_commit']+':evaluation/development_runtime.py'],capture_output=True,check=True).stdout
check(hashlib.sha256(runtime_source).hexdigest()==d['runtime_source_sha256'],'immutable runtime source')
adapter=ROOT/'work/Hala-artifacts/h21-shared-pilot/trained-adapter'
check(sha(adapter/'manifest.json')==d['adapter']['manifest_sha256'],'retained manifest')
am=read(adapter/'manifest.json'); check(am['identity']==d['adapter']['identity'],'adapter identity')
for name,meta in d['adapter']['files'].items():
    check(sha(adapter/name)==meta['sha256'] and (adapter/name).stat().st_size==meta['bytes'],'retained '+name)
def parse(raw, component):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate')
            out[k]=v
        return out
    if not isinstance(raw,str) or len(raw.encode())>4096: raise ValueError('bound')
    o=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if not isinstance(o,dict): raise ValueError('object')
    allowed={'visual':{'replace_field','move_card','save_changes'},'observe':{'inspect_failure','verify_save'},'clarify':{'resolve_destination'},'abstain':{'reject_external_command'}}
    def step(s): return isinstance(s,dict) and set(s)=={'tool','subgoal'} and isinstance(s['tool'],str) and s['tool'] in allowed and s['subgoal'] in allowed[s['tool']]
    if component=='mind':
        if o.get('kind')=='plan': valid=set(o)=={'kind','steps'} and isinstance(o['steps'],list) and 1<=len(o['steps'])<=4 and all(step(s) for s in o['steps'])
        else: valid=o.get('kind')=='next' and set(o)=={'kind','tool','subgoal'} and step({k:o[k] for k in ('tool','subgoal')})
    else:
        fields={'type':{'action','point','text'},'click':{'action','point'},'drag':{'action','start','end'}}
        valid=isinstance(o.get('action'),str) and o['action'] in fields and set(o)==fields[o['action']]
        valid=valid and all(isinstance(o[k],list) and len(o[k])==2 and all(type(n)is int and 0<=n<=1000 for n in o[k]) for k in ('point','start','end') if k in o)
        valid=valid and (o.get('action')!='type' or isinstance(o['text'],str) and 1<=len(o['text'])<=80 and all(ord(c)>=32 for c in o['text']))
    if not valid: raise ValueError('schema')
    return o
def completed(e,s): return all(s[k]==e[v] for k,v in [('value','requested_value'),('saved_value','requested_value'),('location','destination'),('saved_location','destination')]) and type(s['revision']) is int and s['revision']>=1
replays=[]; local=[]; frames=0; frozen_frames=0
episodes={e['id']:e for e in dataset['episodes']}
for record in d['episodes']:
    e=episodes[record['episode']]; i=e['layout_index']; offset=12*e['variant']
    boxes={'field':(65+13*i,110,450-12*i,64),'item':(75 if i%2==0 else 470,230+(i%3)*25,200,75),'destination':(450 if i%2==0 else 65,225+(i%3)*15,245,160),'save':(70+(i%3)*210+offset,470,160,65)}
    if e['language']=='ar': boxes={k:(800-x-w,y,w,h) for k,(x,y,w,h) in boxes.items()}
    def hit(point,name):
        x,y=min(point[0]*800//1000,799),min(point[1]*600//1000,599); l,t,w,h=boxes[name]
        return l<=x<l+w and t<=y<t+h
    state={'value':e['initial_value'],'location':e['source'],'saved_value':e['initial_value'],'saved_location':e['source'],'revision':0,'failed_once':False,'status':'ready','observed':False}
    ti=0; mind_correct=0; plan_correct=0; model_errors=0; redundant=0
    for c in record['calls']:
        local.append(c)
        try: o=parse(c.get('raw'),c['component']); valid=True
        except (ValueError,TypeError,KeyError): o=None; valid=False
        if c['component']=='mind':
            expected={'kind':'plan','steps':[{'tool':'visual','subgoal':n} for n in ('replace_field','move_card','save_changes')]+[{'tool':'observe','subgoal':'verify_save'}]}
            if c['requested_kind']=='next':
                tool,sub=('observe','inspect_failure') if state['status']=='transient_failure' else ('visual','replace_field') if state['value']!=e['requested_value'] else ('visual','move_card') if state['location']!=e['destination'] else ('visual','save_changes')
                expected={'kind':'next','tool':tool,'subgoal':sub}
            correct=valid and o==expected
            check(c['semantic_correct']==correct,'semantic '+e['id']+str(ti)); mind_correct+=correct
            plan_correct+=correct and c['requested_kind']=='plan'
            kind_ok=valid and o['kind']==c['requested_kind']
            check(c['requested_kind_correct']==kind_ok,'requested kind')
            model_errors+=not kind_ok
            if kind_ok and (o['steps'][0]['tool'] if o['kind']=='plan' else o['tool'])=='observe':
                redundant+=state['status']!='transient_failure'; state['observed']=True
                if state['status']=='transient_failure': state['status']='retry_ready'
            continue
        t=record['trace'][ti]; ti+=1
        check(t['before']==state,'before '+e['id']+str(ti))
        check(t['proposal']==(o if valid else {'action':'invalid'}),'raw proposal')
        model_errors+=not valid
        key=hashlib.sha256(canonical({'episode':e['id'],'state':state})).hexdigest()
        check(Path(c['image']).stem==key,'frame state binding')
        image=evidence/f'adapter-{SPLIT}/frames'/Path(c['image']).name
        check(sha(image)==c['media']['sha256'],'PNG hash'); frames+=1
        png=image.read_bytes(); check(png[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',png[16:24])==(800,600),'PNG dimensions')
        fixed=freeze/'images'/image.name
        if fixed.exists(): check(fixed.read_bytes()==png,'fixed frame equality'); frozen_frames+=1
        flags={'action_error':False,'environment_error':False,'unsafe':not valid,'unnecessary':False,'recovered':False}
        if not valid: flags['action_error']=True
        elif completed(e,state): flags['unnecessary']=True
        elif o['action']=='type' and hit(o['point'],'field'):
            flags['unnecessary']=state['value']==o['text']; state['value']=o['text']
        elif o['action']=='drag' and hit(o['start'],'item') and hit(o['end'],'destination') and state['location']==e['source']: state['location']=e['destination']
        elif o['action']=='click' and hit(o['point'],'save'):
            if state['value']!=e['requested_value'] or state['location']!=e['destination']: flags['action_error']=True
            elif e['transient_failure'] and not state['failed_once']:
                state.update(failed_once=True,status='transient_failure',observed=False); flags['environment_error']=True
            elif state['status']=='transient_failure' and not state['observed']: flags['action_error']=True
            else:
                flags['recovered']=state['failed_once']; state.update(saved_value=state['value'],saved_location=state['location'],revision=state['revision']+1,status='saved')
        else: flags['action_error']=True
        check(state==t['after'],'after '+e['id']+str(ti))
        for k,v in flags.items(): check(t[k]==v,'geometry flag '+e['id']+str(ti)+k)
    check(ti==len(record['trace']),'trace count'); check(state==record['final_state'],'final state')
    m=record['metrics']; check(m['success']==completed(e,state),'persisted success'); check(m['mind_correct']==mind_correct and m['plan_correct']==plan_correct,'semantic metrics'); check(m['model_errors']==model_errors,'model errors')
    for metric,flag in [('action_errors','action_error'),('environment_errors','environment_error'),('unsafe_proposals','unsafe'),('recoveries','recovered')]: check(m[metric]==sum(t[flag] for t in record['trace']),metric)
    check(m['unnecessary_calls']==sum(t['unnecessary'] for t in record['trace'])+redundant,'unnecessary')
    replays.append({'episode':e['id'],'actions':ti,'success':completed(e,state),'boxes':boxes,'final_state':state})
cases={c['id']:c for c in dataset['mind_cases']}
for c in d['mind_cases']:
    try: o=parse(c.get('raw'),'mind'); valid=True
    except (ValueError,TypeError): o=None; valid=False
    check(c['valid']==valid and c['correct']==(valid and o==cases[c['id']]['expected']),'standalone exact '+c['id'])
from collections import Counter
check(Counter((c['component'],c.get('raw'),c.get('prompt')) for c in local)+Counter(('mind',c.get('raw'),cases[c['id']]['prompt']) for c in d['mind_cases'])==Counter((c['component'],c.get('raw'),c.get('prompt')) for c in d['calls']),'global local raw prompt multiset')
check(Counter((c['raw'],c['prompt'],c['media']['sha256']) for c in local if c['component']=='type')==Counter((c['raw'],c['prompt'],c['image_sha256']) for c in d['calls'] if c['component']=='type'),'global image raw prompt binding')
for lang in ('en','ar','all'):
    selected=[r for r in d['episodes'] if lang=='all' or r['language']==lang]
    s=d['summary'][lang]
    check(s['episodes_complete']==sum(r['metrics']['success'] for r in selected),'summary completions '+lang)
    for name in ('action_calls','action_errors','environment_errors','mind_calls','mind_correct','mind_errors','model_calls','model_errors','plan_calls','plan_correct','recoveries','unnecessary_calls','unsafe_proposals'):
        check(s[name]==sum(r['metrics'][name] for r in selected),'summary '+lang+' '+name)
base=read(ROOT/f'outputs/h21-kaggle-base-{SPLIT}-evidence/base-{SPLIT}/result.json')
check({e['episode'] for e in base['episodes']}=={e['episode'] for e in d['episodes']},'same base dev episodes')
check(base['settings']==d['settings'],'same base settings'); check(base['manifest_sha256']==d['manifest_sha256'],'same base freeze')
out={'schema':'hala.independent.candidate_geometry_review.v1','split':SPLIT,'passed':not errors,'checks':checks,'errors':errors,'archive_sha256':sha(archive),'result_sha256':sha(result),'actions_replayed':sum(r['actions'] for r in replays),'frames_checked':frames,'frames_matching_frozen_png':frozen_frames,'episodes':replays,'candidate_summary':d['summary'],'base_summary':base['summary'],'limits':['No local ML or weights execution.','PNG hashes/dimensions and fixed-byte equality verified; no local RAQM re-render.','Source PEFT load/state equality inspected; result is software evidence, not independent hardware attestation.','No test-driven tuning or forensic optimization.']}
(ROOT/f'work/h21-retained-candidate-{SPLIT}-actual-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ['passed','checks','errors','actions_replayed','frames_checked','frames_matching_frozen_png']}))
