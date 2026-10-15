import subprocess,os,pathlib,json
cmd=["/usr/bin/python3","-m","training.scripts.train_development","--bundle","/tmp/hala-h21-frozen","--manifest-sha256","ced87ae6874df6de2ed1ab853312824084191a29a42b97869f6eb462aceb629e","--baseline","/tmp/hala-h21-public-baseline-dev.json","--baseline-sha256","e9f98694089d5411d3a6c570c7de2ede8f82cbeb3132e47904d17301030c0768","--source-commit","5b6eb4bf29acbaf21aa199f113f78af4f268b3b7","--probe","/tmp/hala-h21-pretraining-downloaded","--probe-sha256","4d0f824ecf045aadcda97e0323f0dd0aad5451544237d8140cf7725b4408ec1a","--probe-commit","c117f1d30e43aadcb4e9842f421c9029141a584e","--repository","TheHalfMoon/Hala","--output","/tmp/hala-h21-shared-training"]
training_env=dict(os.environ,HALA_REMOTE_DEVELOPMENT_ALLOWED='1',HALA_REMOTE_TRAINING_ALLOWED='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
log=pathlib.Path('/tmp/hala-h21-shared-training.log')
assert not log.exists() and not pathlib.Path('/tmp/hala-h21-shared-training').exists()
with log.open('x') as stream:
    completed=subprocess.run(cmd,cwd='/content/hala-model-development',env=training_env,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
print('HALA_TRAINING_PROCESS:'+json.dumps({'returncode':completed.returncode}),flush=True)
if completed.returncode: raise SystemExit(completed.returncode)
