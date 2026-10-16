import os,json,time,sys,hashlib,subprocess,shutil
from pathlib import Path
root=Path('/kaggle/working/hala-h21'); source=root/'source'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='07591f109b6683c9302fcf890259ddf0bde91494'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=source,text=True).strip()
q=json.loads((root/'h22-attempt2-qualification.json').read_text()); assert q['gpu_uuid']=='GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1' and not q['model_loaded']
for k in ('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','XDG_CACHE_HOME','TORCH_HOME'): os.environ[k]='/tmp/hala-h22-cache'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['DO_NOT_TRACK']='1'
sys.path.insert(0,str(source))
from evaluation.development_precision import check_cache,disk_bytes
check_cache('/tmp/hala-h22-cache'); before=disk_bytes('/tmp/hala-h22-cache')
assert shutil.disk_usage('/tmp').free>6*1024**3
from huggingface_hub import snapshot_download
start=time.time(); path=snapshot_download('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',cache_dir='/tmp/hala-h22-cache',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','*.tiktoken'],max_workers=4)
check_cache('/tmp/hala-h22-cache')
record={'schema':'hala.h22.attempt2.cache_prewarm.v1','source_commit':q['source_commit'],'gpu_uuid':q['gpu_uuid'],'model_revision':'89644892e4d85e24eaac8bacfd4f463576704203','cache_bytes_before':before,'cache_bytes_after':disk_bytes('/tmp/hala-h22-cache'),'wall_seconds':time.time()-start,'files':{str(p.relative_to(path)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(Path(path).rglob('*')) if p.is_file()},'model_loaded':False,'training_runs':0,'scope':'Network preparation only; all later profiles start with disk cache prewarmed; OS page cache and fixed temporal order may differ'}
(root/'h22-attempt2-prewarm.json').write_text(json.dumps(record,sort_keys=True)+'\n')
print('HALA_H22_ATTEMPT2_PREWARM:'+json.dumps(record,sort_keys=True),flush=True)
