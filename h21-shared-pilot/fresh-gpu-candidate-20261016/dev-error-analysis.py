"""DEV-only descriptive error accounting; no production imports or edits."""
import json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(p): return json.loads((root/p).read_text(encoding='utf-8'))
d=read('outputs/h21-kaggle-retry1-adapter-dev-evidence/adapter-dev/result.json')
replay=read('work/h21-retained-candidate-dev-actual-review.json')
dataset=read('outputs/h21-frozen-data/dataset.json')
frozen={e['id']:e for e in dataset['episodes'] if e['split']=='dev'}
boxes={e['episode']:e['boxes'] for e in replay['episodes']}
counts=Counter(); rows=[]; semantic=Counter(); model=Counter(); text=Counter()
for ep in d['episodes']:
    e=frozen[ep['episode']]; b=boxes[e['id']]; language=e['language']
    def hit(p,key):
        x,y=min(p[0]*800//1000,799),min(p[1]*600//1000,599); l,t,w,h=b[key]
        return l<=x<l+w and t<=y<t+h
    per=Counter()
    for t in ep['trace']:
        p=t['proposal']; action=p['action']; before=t['before']
        counts[(language,'actions',action)]+=1
        if action=='type': text['correct' if p['text']==e['requested_value'] else 'incorrect']+=1
        if not t['action_error']: continue
        counts[(language,'errors',action)]+=1
        if action=='type': category='field_point_miss_correct_text' if p['text']==e['requested_value'] else 'field_point_miss_wrong_text'
        elif action=='drag': category='repeat_drag_after_destination_reached'
        elif hit(p['point'],'save'): category='premature_save_target_hit_before_move'
        else: category='save_target_miss_and_premature_save_before_move'
        per[category]+=1
        rows.append({'episode':e['id'],'language':language,'action':action,'category':category,'field_hit':hit(p['point'],'field') if action=='type' else None,'save_hit':hit(p['point'],'save') if action=='click' else None,'item_start_hit':hit(p['start'],'item') if action=='drag' else None,'destination_end_hit':hit(p['end'],'destination') if action=='drag' else None,'already_at_destination':before['location']==e['destination'],'state_changed':t['before']!=t['after']})
    for c in ep['calls']:
        if c['component']=='mind' and not c['semantic_correct']:
            semantic[(language,'invalid_tool_subgoal_pair' if c.get('error') else 'valid_wrong_state_subgoal')]+=1
        if c.get('error'): model[(language,c['component'],c['error'])]+=1
out={'schema':'hala.dev_only.error_analysis.v1','split':'dev','scope':'Eight DEV episodes only; standalone test-labelled probes and TEST failures excluded.','integrity_replay_passed':replay['passed'],'counts':[{'language':k[0],'measure':k[1],'action':k[2],'count':v} for k,v in sorted(counts.items())],'primary_error_categories':dict(Counter(r['category'] for r in rows)),'mind_errors':[{'language':k[0],'category':k[1],'count':v} for k,v in sorted(semantic.items())],'model_errors':[{'language':k[0],'component':k[1],'error':k[2],'count':v} for k,v in sorted(model.items())],'type_text_counts':dict(text),'action_error_records':rows,'future_recommendations_basis':'DEV-only, descriptive proposals; no new pilot or retraining authorized or performed.'}
assert len(rows)==42 and sum(semantic.values())==33 and sum(model.values())==5
assert all(not r['state_changed'] for r in rows)
(root/'work/h21-candidate-dev-error-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('counts','primary_error_categories','mind_errors','model_errors','type_text_counts')}))
