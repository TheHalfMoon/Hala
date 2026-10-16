"""Independent stdlib-only H22 saved-record audit; never runs ML/production code.

Requires an actually retrieved result and external ZIP/result identities. This
cannot turn a partial run, matching metadata, or CPU tests into qualification.
"""
import argparse, copy, hashlib, json, math, struct, subprocess, tempfile, zipfile
from pathlib import Path
from h21_actual_baseline_audit import audit as record_audit, strict, parsed

ROOT=Path(__file__).resolve().parents[1]
SOURCE='07591f109b6683c9302fcf890259ddf0bde91494'
RUNTIME='20b3bf32ca37107ed05e9f2c3ed74413653b51107f5abf474f4a40b25ef58895'
PRECISION='1fb95afef7254b73f7229ec0da5e6ad6d2e91b42f42e5b4e0476f56b75acd9f5'
FREEZE='98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b'
MANIFEST='ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e'
ADAPTER='7e46995aa694e5782a93d4beb6139cc5880b898a2038eb2d2ce2a2d8006a841d'
UUID='GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1'
ORDER=[('fp16','dev'),('int8','dev'),('nf4','dev'),('fp16','test'),('int8','test'),('nf4','test')]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def number(v): return type(v) in (int,float) and math.isfinite(v) and v>=0
def byte_count(v): return type(v) is int and v>=0
def same(a,b): return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def complete(e,s): return type(s.get('revision')) is int and s['revision']>0 and s.get('value')==s.get('saved_value')==e['requested_value'] and s.get('location')==s.get('saved_location')==e['destination']

def audit(args):
    result_path=Path(args.result); raw=result_path.read_bytes(); d=strict(raw)
    errors=[]; warnings=[]; checks=0
    def check(condition,label):
        nonlocal checks
        checks+=1
        if not condition: errors.append(label)
    check(sha(raw)==args.result_sha256,'external result SHA')
    archive=Path(args.archive); check(sha(archive.read_bytes())==args.archive_sha256,'external ZIP SHA')
    with zipfile.ZipFile(archive) as z:
        check(z.testzip() is None,'ZIP CRCs')
        members=[n for n in z.namelist() if not n.endswith('/')]
        check('run/result.json' in members and z.read('run/result.json')==raw,'ZIP result binding')
        for n in members:
            target=(result_path.parent.parent/n).resolve()
            check(target.is_relative_to(result_path.parent.parent.resolve()),'safe archive name')
            if target.is_relative_to(result_path.parent.parent.resolve()): check(target.is_file() and target.read_bytes()==z.read(n),'extracted bytes '+n)
    check(d.get('schema')=='hala.development.precision.v1','H22 schema')
    check(d.get('reference_schema')=='hala.development.adapter_evaluation.v1','fixed retained-adapter reference')
    check(d.get('status')=='measured','complete measured run required')
    for flag in ('adoption_qualified','quality_qualified','native_qualified','execution_authorized'):
        check(d.get(flag) is False,flag+' remains false')
    check(d.get('adapter_loaded') is True,'adapter software loaded')
    check(d.get('source_commit')==SOURCE,'source commit')
    check(d.get('runtime_source_sha256')==RUNTIME,'runtime source digest')
    repo=Path(args.source_repo)
    def source(name): return subprocess.run(['git','-C',str(repo),'show',SOURCE+':'+name],capture_output=True,check=True).stdout
    check(sha(source('evaluation/development_runtime.py'))==RUNTIME,'historical runtime bytes')
    check(sha(source('evaluation/development_precision.py'))==PRECISION,'historical precision bytes')
    freeze=source('experiments/development-precision/source-freeze.json')
    check(sha(freeze)==FREEZE,'prospective source freeze SHA')
    for name,digest in strict(freeze)['files'].items(): check(sha(source(name))==digest,'source freeze '+name)
    config=strict(source('experiments/development-precision/config.json'))
    selected=d.get('precision_profile',{}); mode,split=ORDER[args.index]
    check(d.get('split')==split,'fixed index split')
    expected=copy.deepcopy(config['profiles'][('fp16','int8','nf4').index(mode)]); expected['source_freeze_sha256']=FREEZE
    check(same(selected,expected),'prespecified exact precision profile')
    check(config.get('adoption_qualified') is False and config.get('quality_qualified') is False,'prospective config cannot adopt')
    bundle=Path(args.bundle); dataset=strict((bundle/'dataset.json').read_bytes())
    adapter=Path(args.adapter); am=strict((adapter/'manifest.json').read_bytes())
    check(sha((adapter/'manifest.json').read_bytes())==ADAPTER,'original adapter manifest')
    check(d.get('adapter',{}).get('manifest_sha256')==ADAPTER,'run adapter manifest')
    check(same(d.get('adapter',{}).get('identity'),am['identity']),'original adapter identity')
    check(same(d.get('adapter',{}).get('files'),am['files']),'original adapter payload inventory')
    for name,m in am['files'].items(): check(sha((adapter/name).read_bytes())==m['sha256'] and (adapter/name).stat().st_size==m['bytes'],'original adapter bytes '+name)
    expected_identity={key:d.get(key) for key in ('source_commit','manifest_sha256','data_file_sha256','frozen_source','model_id','model_revision','split','settings','evaluation_provider','runtime_source_sha256','packages','python','cuda','gpu','gpu_capability','gpu_uuid')}
    expected_identity.update(source_freeze_sha256=FREEZE,adapter=d.get('adapter'),reference_settings=selected.get('reference_settings'),tasks=[e['id'] for e in dataset['episodes'] if e['split']==split],mind_tasks=[c['id'] for c in dataset['mind_cases']])
    check(same(d.get('comparison_identity'),expected_identity),'comparison identity exact field/binding equality')
    check([e['episode'] for e in d['episodes']]==expected_identity['tasks'],'ordered episode identity')
    check([c['id'] for c in d['mind_cases']]==expected_identity['mind_tasks'],'ordered repeated standalone identity')
    check(d.get('gpu_uuid')==UUID,'attempt2 physical UUID record')
    pins={'torch':'2.11.0+cu128','transformers':'5.16.1','Pillow':'11.3.0','peft':'0.20.0','safetensors':'0.8.0','accelerate':'1.12.0','bitsandbytes':'0.50.2'}
    check(d.get('packages')==pins and d.get('imported_packages')==pins,'exact distribution/import package pins')
    # Schema-only projection lets the already independent H21 checker rescore
    # unchanged H19 records. Original JSON is untouched and SHA-bound above.
    projected=copy.deepcopy(d); projected['schema']=d.get('reference_schema')
    with tempfile.TemporaryDirectory(prefix='hala-h22-independent-') as temp:
        p=Path(temp)/'reference.json'; p.write_text(json.dumps(projected,allow_nan=False),encoding='utf-8')
        underlying=record_audit(p,bundle,MANIFEST)
    checks+=underlying['checks']; errors.extend('closed-loop record: '+e for e in underlying['errors'])
    geometry_actions=0; png_frames=0; fixed_png_matches=0
    episodes={e['id']:e for e in dataset['episodes']}
    for ep in d['episodes']:
        e=episodes[ep['episode']]; i=e['layout_index']; offset=e['variant']*12
        boxes={'field':(65+13*i,110,450-12*i,64),'item':(75 if i%2==0 else 470,230+(i%3)*25,200,75),'destination':(450 if i%2==0 else 65,225+(i%3)*15,245,160),'save':(70+(i%3)*210+offset,470,160,65)}
        if e['language']=='ar': boxes={k:(800-x-w,y,w,h) for k,(x,y,w,h) in boxes.items()}
        def hit(point,name):
            x,y=min(point[0]*800//1000,799),min(point[1]*600//1000,599); l,t,w,h=boxes[name]
            return l<=x<l+w and t<=y<t+h
        state=dict(value=e['initial_value'],location=e['source'],saved_value=e['initial_value'],saved_location=e['source'],revision=0,failed_once=False,status='ready',observed=False)
        ti=0
        for call in ep['calls']:
            answer=parsed(call['component'],call.get('raw'))
            if call['component']=='mind':
                if answer and answer['kind']==call['requested_kind']:
                    tool=answer['steps'][0]['tool'] if answer['kind']=='plan' else answer['tool']
                    if tool=='observe':
                        state['observed']=True
                        if state['status']=='transient_failure': state['status']='retry_ready'
                continue
            t=ep['trace'][ti]; ti+=1; geometry_actions+=1
            check(t['before']==state,'geometry before-state '+e['id'])
            frame=result_path.parent/'frames'/Path(call['image']).name
            key=sha(canonical({'episode':e['id'],'state':state}))
            check(frame.stem==key,'frame private state digest binding')
            data=frame.read_bytes(); check(sha(data)==call['media']['sha256'],'PNG content SHA'); png_frames+=1
            check(data[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',data[16:24])==(800,600),'PNG signature/dimensions')
            check(call['media']['font_sha256']=='ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280' and call['media']['raqm']=='0.10.1','qualified media font/RAQM metadata')
            fixed=bundle/'images'/frame.name
            if fixed.exists(): check(data==fixed.read_bytes(),'fixed external PNG equality'); fixed_png_matches+=1
            flags=dict(action_error=False,environment_error=False,unsafe=answer is None,unnecessary=False,recovered=False)
            if answer is None: flags['action_error']=True
            elif complete(e,state): flags['unnecessary']=True
            elif answer['action']=='type' and hit(answer['point'],'field'):
                flags['unnecessary']=state['value']==answer['text']; state['value']=answer['text']
            elif answer['action']=='drag' and hit(answer['start'],'item') and hit(answer['end'],'destination') and state['location']==e['source']: state['location']=e['destination']
            elif answer['action']=='click' and hit(answer['point'],'save'):
                if state['value']!=e['requested_value'] or state['location']!=e['destination']: flags['action_error']=True
                elif e['transient_failure'] and not state['failed_once']:
                    state.update(failed_once=True,status='transient_failure',observed=False); flags['environment_error']=True
                elif state['status']=='transient_failure' and not state['observed']: flags['action_error']=True
                else:
                    flags['recovered']=state['failed_once']; state.update(saved_value=state['value'],saved_location=state['location'],revision=state['revision']+1,status='saved')
            else: flags['action_error']=True
            check(state==t['after'],'geometry after-state '+e['id'])
            for k,v in flags.items(): check(t[k] is v,'geometry flag '+k)
        check(state==ep['final_state'],'complete final-state chain '+e['id'])
    telemetry=d.get('telemetry_errors',{})
    memory=[]
    def memory_row(row,label):
        if row is None:
            warnings.append('Unavailable memory telemetry: '+label); return
        keys=('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes','device_used_bytes','device_total_bytes','cpu_rss_bytes','cpu_peak_rss_bytes')
        for key in keys:
            if row.get(key) is None: warnings.append('Null memory field: '+label+'/'+key)
            else: check(byte_count(row[key]),'nonnegative integer byte unit '+label+'/'+key)
        for left,right in [('allocated_bytes','reserved_bytes'),('allocated_bytes','peak_allocated_bytes'),('reserved_bytes','peak_reserved_bytes'),('peak_allocated_bytes','peak_reserved_bytes'),('device_used_bytes','device_total_bytes')]:
            if row.get(left) is not None and row.get(right) is not None: check(row[left]<=row[right],'memory ordering '+label+'/'+left)
        memory.append(row)
    for key in ('memory_before_load','memory_loaded','memory_final'):
        memory_row(d.get(key),key)
        if d.get(key) is None: check(key in telemetry,'top-level missing-memory error explanation '+key)
    for c in d['calls']:
        for key in ('memory_before','memory_after'): memory_row(c.get(key),str(c['call_index'])+'/'+key)
        for key in ('wall_seconds','preprocessing_seconds','generation_seconds'):
            if key in c: check(number(c[key]),'finite nonnegative call seconds '+key)
        if all(number(c.get(k)) for k in ('wall_seconds','preprocessing_seconds','generation_seconds')):
            check(c['wall_seconds']+0.01>=c['preprocessing_seconds']+c['generation_seconds'],'call wall includes preprocessing/generation')
    used=[r['device_used_bytes'] for r in memory if byte_count(r.get('device_used_bytes'))]
    check(d.get('max_sampled_device_used_bytes')==(max(used) if used else None),'sampled CUDA used-byte maximum arithmetic')
    for key in ('load_seconds','wall_seconds','cpu_process_seconds','model_generation_seconds'): check(number(d.get(key)),'finite nonnegative run seconds '+key)
    generation=sum(c.get('generation_seconds',0) for c in d['calls'])
    check(math.isclose(d['model_generation_seconds'],generation,rel_tol=1e-12,abs_tol=1e-6),'sum generation seconds')
    check(d['wall_seconds']+0.1>=d['load_seconds']+generation,'run wall includes load and generation')
    cache=d.get('unique_physical_cache_bytes')
    check(byte_count(cache) or cache is None and 'cache_bytes' in telemetry,'cache content bytes or explicit unavailable error')
    residency=d.get('residency_loaded')
    if residency is None: check(False,'measured precision requires residency record')
    else:
        tensors=residency.get('tensors',[]); check(bool(tensors),'nonempty full-model residency')
        names=[t['name'] for t in tensors]; check(len(names)==len(set(names)),'unique residency tensor names')
        for t in tensors:
            check(t['device']=='cuda:0' and byte_count(t['storage_bytes_unique']),'residency device/unique byte metadata')
            check(t['vision'] is ('vision' in t['name'] or 'visual' in t['name']),'vision subset classifier')
        check(residency['all_storage_bytes']==sum(t['storage_bytes_unique'] for t in tensors),'all storage byte sum')
        check(residency['vision_storage_bytes']==sum(t['storage_bytes_unique'] for t in tensors if t['vision'])>0,'vision storage byte sum')
        if mode in ('int8','nf4'):
            quant_dtype='torch.int8' if mode=='int8' else 'torch.uint8'
            check(any(t['dtype']==quant_dtype and t['storage_bytes_unique']>0 for t in tensors),'actual quantized storage dtype evidence')
            check(any(t['dtype']==quant_dtype and t['vision'] and t['storage_bytes_unique']>0 for t in tensors),'actual vision quantized storage evidence')
    process=d.get('process_cost'); samples=[]
    if process is None:
        check('process_cost' in telemetry or 'process_cost_start' in telemetry,'unavailable process sampler explicit error'); warnings.append('Process sampler unavailable')
    else:
        check(process.get('continuous_peak_qualified') is False,'sampled observer never continuous peak')
        check(process.get('minimum_wait_between_samples_seconds')==2,'prespecified sampler interval')
        check(type(process.get('sample_limit')) is int and 1<=process['sample_limit']<=1802,'bounded sampler limit')
        samples=process.get('samples',[]); check(len(samples)<=process['sample_limit'],'bounded saved samples')
        previous=None
        for s in samples:
            if 'error_type' in s: warnings.append('Sampler row error: '+s['error_type']); continue
            check(number(s.get('monotonic_seconds')) and number(s.get('cpu_process_seconds')),'sample clocks seconds')
            if previous:
                check(s['monotonic_seconds']>=previous['monotonic_seconds'] and s['cpu_process_seconds']>=previous['cpu_process_seconds'],'monotonic sample clocks')
            previous=s
            if s.get('cpu_rss_bytes') is None: check('rss_error' in s,'RSS unavailable reason')
            else: check(byte_count(s['cpu_rss_bytes']),'RSS byte units')
            if s.get('device_uuid') is None: check(s.get('device') is None and s.get('device_error')=='DeviceIdentityUnavailable','early unresolved UUID cannot imply device sample')
            else: check(s['device_uuid']==UUID,'sampler physical UUID consistency')
            device=s.get('device')
            if device is None: check(bool(s.get('device_error')),'device unavailable reason'); continue
            check(set(device)=={'used_mib','free_mib','utilization_percent','power_watts'},'device sample field/unit set')
            for key,value in device.items():
                if value is None: check(key in s.get('device_field_errors',{}),'unsupported value explicit reason')
                else: check(number(value),'finite nonnegative device metric '+key)
            if device.get('utilization_percent') is not None: check(device['utilization_percent']<=100,'GPU utilization percent range')
    return dict(schema='hala.independent.h22.precision_record_audit.v1',passed=not errors,checks=checks,errors=errors,warnings=warnings,result_sha256=sha(raw),archive_sha256=args.archive_sha256,profile_index=args.index,mode=mode,split=split,result_status=d.get('status'),complete_measured_run=d.get('status')=='measured' and not errors,independent_summary=underlying['independent_summary'],geometry_actions_replayed=geometry_actions,png_frames_checked=png_frames,fixed_png_matches=fixed_png_matches,process_samples=len(samples),physical_uuid_metadata_consistent=d.get('gpu_uuid')==UUID,adoption_qualified=False,quality_qualified=False,native_qualified=False,execution_authorized=False,limitations=['Saved-record consistency only; no independent hardware/run attestation.','Schema-only H21 reference projection for independent closed-loop scoring; original JSON untouched and SHA-bound.','PNG hashes/state binding and fixed image equality only; no independent RAQM rerender of newly reached states.','Storage/cache sums do not independently prove pointer/inode deduplication.','Bytes are bytes; MiB values use 1048576 bytes, watts are sampled power, not joules or money. No missing metric becomes zero.','Same physical UUID metadata cannot remove fixed-order/warm-cache confounding; no equal-accuracy/adoption/release claim.','Standalone test-labelled probes are repeated exposed diagnostics, never new holdout evidence.'])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('result'); p.add_argument('--archive',required=True); p.add_argument('--result-sha256',required=True); p.add_argument('--archive-sha256',required=True); p.add_argument('--index',type=int,choices=range(6),required=True)
    p.add_argument('--bundle',default=str(ROOT/'outputs/h21-frozen-data')); p.add_argument('--adapter',default=str(ROOT/'work/Hala-artifacts/h21-shared-pilot/trained-adapter')); p.add_argument('--source-repo',default=str(ROOT/'work/Hala-model')); p.add_argument('--output',required=True); args=p.parse_args()
    try: report=audit(args)
    except Exception as error:
        report=dict(schema='hala.independent.h22.precision_record_audit.v1',passed=False,complete_measured_run=False,adoption_qualified=False,quality_qualified=False,error_type=type(error).__name__,error=str(error),scope='Audit incomplete; no actual counts or qualification inferred')
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:report[k] for k in ('passed','checks','errors','error_type','geometry_actions_replayed','process_samples') if k in report})); raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__': main()
