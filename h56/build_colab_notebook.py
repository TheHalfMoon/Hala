"""Build one clickable genuine native-drag baseline Colab notebook, no model run."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"h56/colab-h56-actual-qwen-drag-baseline.ipynb"
MODEL_SCRIPT=ROOT/"h56/real_drag_baseline.py"
def cell(kind,source):
    return dict(cell_type=kind,metadata={},
      **({"source":[line+"\n" for line in source.strip().split("\n")],"outputs":[],"execution_count":None}
         if kind=="code" else {"source":[line+"\n" for line in source.strip().split("\n")]}))

def build():
    if OUT.exists():raise FileExistsError("Create-only H56 Colab notebook")
    sha=hashlib.sha256(MODEL_SCRIPT.read_bytes()).hexdigest()
    nb=dict(nbformat=4,nbformat_minor=5,metadata={
       "colab":{"provenance":[],"name":"H56 Actual Qwen3-VL-2B Ordered-Drag Baseline","gpuType":"T4"},
       "kernelspec":{"name":"python3","display_name":"Python 3"},
       "language_info":{"name":"python"}},
       cells=[
        cell("markdown",'''# Hala H56 — Real Qwen3-VL-2B ordered-drag baseline on FREE interactive Colab

**Purpose:** Run **ACTUAL unchanged Qwen3-VL-2B** (revision `89644892e4d85e24eaac8bacfd4f463576704203`) on the 24 genuine first-party H55 public-DEV screenshots (12 Arabic, 12 English). No H Company TRAIN screenshot source is needed. The notebook never opens H55 golden `capture.json` while asking the model questions, never trains a checkpoint and never creates fabricated model results.

**Costs and restrictions:** Use **Google Colab's FREE GPU tier only**. In Colab choose **Runtime → Change runtime type → T4 GPU** (or another free GPU offered to your session), then **Runtime → Run all**. Colab GPU availability is not guaranteed. If the free tier offers no GPU, stop; **do not purchase compute units or Colab Pro**. This notebook uses the interactive Colab notebook UI, not SSH, tunnels, or background remote control. It will download several gigabytes of open model weights **into the remote ephemeral Colab workspace**, not your PC. No paid API, no Kaggle GPU compute, no HF paid Jobs. Hala train-source weights and credentials are not needed.

**Research boundary:** H55 is PUBLIC DEV, not independently blind TEST. Even 24/24 real model success here would NOT prove that Hala beats any competitor. Results only become a verified GUI task-success baseline after copying the unchanged raw outputs into Hala's audited Windows Edge replayer. If no GPU, notebook should exit without ever labelling model inference complete.'''),
        cell("code",'''import os, platform, subprocess, sys
assert 'COLAB_RELEASE_TAG' in os.environ, 'Open in interactive Google Colab, not another paid job or local PC'
assert platform.system() == 'Linux', 'Remote Linux runtime only'
assert 'KAGGLE_KERNEL_RUN_TYPE' not in os.environ, 'Kaggle GPU runtime is not authorized'
subprocess.run(['nvidia-smi'],check=True)
import torch
assert torch.cuda.is_available(), 'No FREE Colab CUDA GPU available. Stop; never purchase paid GPU.'
p=torch.cuda.get_device_properties(0)
print('ACTUAL_FREE_COLAB_DEVICE',p.name,'VRAM_GB',round(p.total_memory/1e9,2))
print('Colab free-tier GPU time is dynamic and not guaranteed. This is an INTERACTIVE notebook run.')'''),
        cell("code",'''import subprocess,sys
# Model weights and processor run only after free GPU has been confirmed.
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.16.1','Pillow>=11,<13',
                'accelerate>=1,<3','safetensors>=0.5,<1'],check=True)
import transformers
assert transformers.__version__ == '5.16.1', transformers.__version__
print('H56_DEPENDENCIES_PINNED',transformers.__version__)'''),
        cell("code",'''from pathlib import Path
import hashlib,subprocess,os
repo=Path('/content/hala-h56')
assert not repo.exists(), 'Create-only checkout: do not overwrite previous experiment.'
subprocess.run(['git','clone','--depth','1','--branch','ps/h54-native-drag-lora',
                'https://github.com/TheHalfMoon/Hala.git',str(repo)],check=True)
actual=hashlib.sha256((repo/'h56/real_drag_baseline.py').read_bytes()).hexdigest()
expected='''+repr(sha)+'''
assert actual==expected, 'Model runner reviewed source bytes do not match pinned notebook; stop.'
from importlib.util import spec_from_file_location,module_from_spec
os.chdir(repo)
import sys
sys.path.insert(0,str(repo))
from h56.real_drag_baseline import source_inputs
assert len(source_inputs('h55/assets'))==24
print('VERIFIED_REVIEWED_H56_RUNNER_SHA256',actual)
print('No model input or GPU trial used H55 source answer oracle capture.json.')'''),
        cell("code",'''import os,datetime
from h56.real_drag_baseline import run
os.environ['HALA_REMOTE_BASELINE_ALLOWED']='1'  # explicit interactive FREE Colab opt-in
run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out='/content/hala-h56-real-qwen-'+run_id
print('BEGIN_ACTUAL_UNCHANGED_QWEN3_VL_2B',out)
report=run('h55/assets',out,max_cases=24)
assert report['status']=='completed'
assert len(report['real_generated_cases'])==24
print('REAL_NEURAL_GENERATIONS',24,'ARABIC',12,'ENGLISH',12,'NO_TRAINING',True)
print('RAW_RESULTS',out+'/raw-attempts.json')
print('DO NOT CLAIM GUI TASK SUCCESS UNTIL REAL EDGE REPLAY.')'''),
        cell("code",'''from pathlib import Path
import shutil
from google.colab import files
folder=Path(out)
assert (folder/'run-result.json').is_file() and (folder/'raw-attempts.json').is_file()
archive=shutil.make_archive(str(folder), 'zip', str(folder))
print('Download actual untouched raw output and original model/gpu/runtime receipt.')
print('Upload the ZIP to ChatGPT so Hala can run h55/score.py and REAL Edge replay on Windows.')
files.download(archive)''')])
    OUT.write_text(json.dumps(nb,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    parsed=json.loads(OUT.read_text("utf8"))
    assert parsed["nbformat"]==4 and len(parsed["cells"])==6
    return dict(notebook=str(OUT),runner_sha256=sha,cells=len(parsed["cells"]),
                contains_real_gpu_execution=False)
if __name__=="__main__":
    print(json.dumps(build(),sort_keys=True))
