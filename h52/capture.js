/* H52: actual local browser screenshot + step/decoy replay of first-party scenes.
 * No real services, cloud, models or paid resources. Network blocked.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{pathToFileURL}=require('url');
const core=process.env.PUPPETEER_CORE_PATH;
if(!core||JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0')
  throw Error('Only pinned puppeteer-core 25.12.0');
const puppeteer=require(core);
const root=path.resolve(process.argv[2]||'.');
if(!fs.existsSync(root)||!fs.existsSync(path.join(root,'source.json')))throw Error('Missing frozen H52 authored source');
for(const name of ['png','capture.json','authoring_sft.jsonl','sft_manifest.json']){
 if(fs.existsSync(path.join(root,name)))throw Error('Create-only output already present: '+name);
}
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const source=JSON.parse(fs.readFileSync(path.join(root,'source.json'),'utf8'));
const plans=JSON.parse(fs.readFileSync(path.join(root,'plans.json'),'utf8'));
if(source.count!==96||plans.length!==96||source.authored_plans_sha256!==sha(fs.readFileSync(path.join(root,'plans.json')))){
 // The source uses canonical indent/newline; digest must match exact frozen file.
 throw Error('Frozen original authored plan changed');
}
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Real Edge browser not installed');
let blocked=0;
(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,
   args:['--no-first-run','--disable-extensions','--disable-background-networking',
         '--disable-sync','--disable-component-update']});
 const captures=[],rows=[],files=[];
 try{
 const page=await browser.newPage();
 await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
 await page.setRequestInterception(true);
 page.on('request',req=>{if(req.url().startsWith('file:'))req.continue();
     else {blocked++;req.abort();}});
 fs.mkdirSync(path.join(root,'png'));
 for(const plan of plans){
   const original=source.scenes.find(r=>r.id===plan.id);
   const bytes=fs.readFileSync(path.join(root,'html',plan.id+'.html'));
   if(!original || sha(bytes)!==original.sha256)throw Error('Original HTML SHA mismatch');
   await page.goto(pathToFileURL(path.join(root,'html',plan.id+'.html')).href,
                   {waitUntil:'load',timeout:20000});
   for(let stage=0;stage<3;stage++){
     const screen=plan.id+'-s'+stage, photo='png/'+screen+'.png';
     const expected=plan.targets[stage],order=plan.orders[stage];
     const controls=await page.evaluate(()=>[...document.querySelectorAll('button.pick')].map(b=>{
        const r=b.getBoundingClientRect();return {key:Number(b.dataset.key),
          rect:[r.x,r.y,r.width,r.height]} }));
     if(controls.length!==8 || new Set(controls.map(r=>r.key)).size!==8 ||
        controls.some(r=>r.rect[2]<60||r.rect[3]<35||r.rect[0]<0||r.rect[1]<0||
        r.rect[0]+r.rect[2]>1024||r.rect[1]+r.rect[3]>625))
          throw Error('Button geometry unbound '+screen);
     if(controls.map(r=>r.key).join()!=order.join())throw Error('DOM order mismatch '+screen);
     const target=controls.find(r=>r.key===expected);
     const [x,y,width,height]=target.rect,point=[Math.round(x+width/2),Math.round(y+height/2)];
     const intersects=controls.filter(r=>point[0]>=r.rect[0]&&point[0]<r.rect[0]+r.rect[2]&&
         point[1]>=r.rect[1]&&point[1]<r.rect[1]+r.rect[3]);
     if(intersects.length!==1||intersects[0].key!==expected)throw Error('Ambiguous target center '+screen);
     const shot=await page.screenshot({type:'png',captureBeyondViewport:false});
     fs.writeFileSync(path.join(root,photo),shot,{flag:'wx'});
     files.push({path:photo,sha256:sha(shot),width:1024,height:640});
     const goal=plan.labels[expected];
     const prefix=plan.lang==='ar'?'أكمل تسلسل الاختيارات في الشاشة التدريبية الخيالية.':'Complete the fictional training workflow.';
     const sequence=plan.targets.map(k=>plan.labels[k]);
     const prompt=prefix+' '+(plan.lang==='ar'?'الترتيب: ':'Order: ')+
       sequence.join(' → ')+'. '+(plan.lang==='ar'?'الخطوة الحالية: ':'Current step: ')+
       (stage+1)+'/3. '+(plan.lang==='ar'?'اختر ':'Select ')+goal+
       '. Return ONLY one JSON window-pixel click proposal. Screenshot 1024x640.';
     const completion=JSON.stringify({type:'click',point,
                 coordinate_space:'window',image_scale:1,element_ref_hint:'visual_target'});
     rows.push({id:screen,split:'train',family:plan.family,language:plan.lang,
             prompt,completion,image_path:photo,image_sha256:sha(shot)});
     let wrongBefore=null,wrongAfter=null;
     if(stage===0 && plan.id.endsWith('00')){
       const decoy=controls.find(r=>r.key!==expected);
       await page.locator('button.pick[data-key="'+decoy.key+'"]').click();
       const current=await page.evaluate(()=>({finished:Number(document.body.dataset.completedSteps||0),
           errors:Number(document.body.dataset.wrong||0)}));
       if(current.finished!==stage||current.errors!==1)
          throw Error('Wrong-target accepted unexpectedly');
       wrongBefore=decoy.key;wrongAfter=current.errors;
     }
     await page.locator('button.pick[data-key="'+expected+'"]').click();
     const status=await page.evaluate(()=>({finished:Number(document.body.dataset.completedSteps||0),
           correct:Number(document.body.dataset.lastCorrect),taskFinished:document.body.dataset.taskFinished==='true'}));
     if(status.finished!==stage+1||status.correct!==expected||(stage===2)!==status.taskFinished)
        throw Error('Real browser postcondition not achieved '+screen);
     captures.push({screen,source_id:plan.id,stage,language:plan.lang,
             screenshot_sha256:sha(shot),original_html_sha256:original.sha256,
             target_key:expected,point,controls,expected_step_after:stage+1,
             observed_step_after:status.finished,wrong_target_tested:wrongBefore!==null,
             wrong_target_key:wrongBefore,wrong_clicks_observed:wrongAfter});
   }
 }
 if(captures.length!==288||rows.length!==288||files.length!==288||
    new Set(files.map(x=>x.sha256)).size!==288)throw Error('Insufficient distinct screenshots');
 const raw=rows.map(r=>JSON.stringify(r)).join('\n')+'\n';
 const rights={origin:'original_first_party_fictional_task_factory',
    license:'CC0-1.0',official_interface:false,external_assets:false,
    personal_data:false,real_accounts:false,benchmark_answers:false};
 const manifest={schema:'hala.training.manifest.v1',
     source_schema:'hala.h52.original-three-step-browser-train.v1',
     split:'train',source_kind:'synthetic',rights_cleared:true,
     token_budget:4096,rights,source_sha256:sha(fs.readFileSync(path.join(root,'source.json'))),
     dataset_sha256:sha(Buffer.from(raw)),unique_workflows:96,
     distinct_original_screenshots:288,train_rows:288,steps_per_workflow:3,
     independent_blind_test:false,dev_or_benchmark_examples:0,
     model_training_executed:false,media:files};
 const receipt={schema:'hala.h52.real-edge-three-step-replay.v1',source_sha256:manifest.source_sha256,
     train_rows:288,independent_screenshot_hashes:288,
     browser:'Microsoft Edge, genuine headless DOM clicks, 1024x640',
     private_data:false,real_service_access:false,model_inference:false,
     provider_data_download:false,external_network_requests_rejected:blocked,
     successful_real_step_clicks:288,wrong_clicks_tested:captures.filter(x=>x.wrong_target_tested).length,
     all_completed_workflows:96,cases:captures};
 fs.writeFileSync(path.join(root,'authoring_sft.jsonl'),raw,{flag:'wx'});
 fs.writeFileSync(path.join(root,'sft_manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 fs.writeFileSync(path.join(root,'capture.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
 console.log('H52_GENUINE_EDGE_WORKFLOWS',96,'SCREENSHOTS',288,
             'SUCCESSFUL_CLICK_STEPS',288,'WRONG_TARGET_REJECTIONS',receipt.wrong_clicks_tested,
             'DATASET_SHA256',manifest.dataset_sha256);
 }finally{await browser.close();}
})().catch(err=>{console.error(err.stack);process.exitCode=1});
