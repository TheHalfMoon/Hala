/* H55: execute supplied actual-model (or explicitly synthetic-unit) drag JSON
 * on real local Edge public DEV scenes. NEVER uses source oracle as prediction.
 * No GPU or model invocation, no live services, no public TEST assertion.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{pathToFileURL}=require('url');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const core=process.env.PUPPETEER_CORE_PATH;
if(!core||JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0')
 throw Error('Expected pinned puppeteer-core 25.12.0');
const puppeteer=require(core);
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Installed genuine Edge required');
const root=path.resolve(process.argv[2]||''),sourceFile=path.resolve(process.argv[3]||'');
const output=path.resolve(process.argv[4]||'');
if(!fs.existsSync(root)||!fs.existsSync(sourceFile)||fs.existsSync(output))
 throw Error('Qualified original source, raw model input, and CREATE-ONLY output required');
const receipt=JSON.parse(fs.readFileSync(path.join(root,'capture.json'),'utf8'));
const source=JSON.parse(fs.readFileSync(path.join(root,'source.json'),'utf8'));
const attempts=JSON.parse(fs.readFileSync(sourceFile,'utf8'));
if(attempts.schema!=='hala.h55.public-dev-raw-drag-attempts.v1'||
   attempts.source_sha256!==receipt.source_sha256||receipt.source_sha256!==sha(fs.readFileSync(path.join(root,'source.json')))||
   !['synthetic_unit_fixture','provider_raw_model_outputs'].includes(attempts.origin)||
   !Array.isArray(attempts.cases)||attempts.cases.length!==24)
 throw Error('Only exactly source-bound 24-case public DEV raw attempts admitted');
const truth=new Map(receipt.cases.map(r=>[r.id,r]));
const map=new Map();
for(const p of attempts.cases){
 if(!p || Object.keys(p).sort().join()!=='id,image_sha256,raw_output'||
    map.has(p.id)||!truth.has(p.id)||truth.get(p.id).image_sha256!==p.image_sha256||
    typeof p.raw_output!=='string'||p.raw_output.length>4096)
   throw Error('Unbound model/fixture output or immutable screenshot');
 map.set(p.id,p.raw_output);
}
function parse(raw){
 let v;try{v=JSON.parse(raw);}catch{return null;}
 if(!v||Array.isArray(v)||Object.keys(v).sort().join()!=='coordinate_space,end_point,ordered,start_point,type'||
    v.type!=='drag'||v.coordinate_space!=='source_image_pixels'||v.ordered!==true)return null;
 for(const key of ['start_point','end_point']){
  const x=v[key];
  if(!Array.isArray(x)||x.length!==2||!x.every((n,i)=>Number.isInteger(n)&&n>=0&&n<(i===0?1024:640)))return null;
 }
 return v;
}
(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
  '--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update']});
 let blocked=0;
 const results=[];
 try{
  const page=await browser.newPage();await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',r=>{if(r.url().startsWith('file:'))r.continue();else{blocked++;r.abort();}});
  for(const p of receipt.cases){
   const htmlPath=path.join(root,'html',p.id+'.html');
   if(sha(fs.readFileSync(htmlPath))!==p.source_html_sha256)throw Error('Mutated original authored DEV HTML');
   const png=path.join(root,p.image_path);
   if(sha(fs.readFileSync(png))!==p.image_sha256)throw Error('Mutated source screenshot');
   const candidate=parse(map.get(p.id));
   await page.goto(pathToFileURL(htmlPath).href,{waitUntil:'load'});
   if(candidate){
    await page.mouse.move(...candidate.start_point);
    await page.mouse.down();
    await page.mouse.move(...candidate.end_point,{steps:12});
    await page.mouse.up();
   }
   const browserStatus=await page.evaluate(()=>({
     accepted:document.body.dataset.accepted==='true',
     complete:Number(document.body.dataset.completed),
     rejected:Number(document.body.dataset.rejected)}));
   const success=browserStatus.accepted&&browserStatus.complete===1;
   results.push({id:p.id,language:p.language,valid_action:!!candidate,
      real_edge_drag_completed:success,observed_status:browserStatus});
  }
 }finally{await browser.close();}
 const langs={};
 for(const language of ['ar','en']){
  const rows=results.filter(x=>x.language===language);
  langs[language]={cases:rows.length,valid:rows.filter(x=>x.valid_action).length,
                   real_edge_success:rows.filter(x=>x.real_edge_drag_completed).length};
 }
 const report={schema:'hala.h55.real-edge-prediction-replay.v1',
   origin:attempts.origin,source_sha256:receipt.source_sha256,attempts_sha256:sha(fs.readFileSync(sourceFile)),
   actual_original_edge_mouse_drag_executed:true,public_dev_not_blind:true,
   synthetic_fixture_only:attempts.origin==='synthetic_unit_fixture',
   independent_model_inference_provenance_verified:false,
   neural_training_executed:false,official_benchmark_comparison_qualified:false,
   external_network_requests_rejected:blocked,by_language:langs,rows:results};
 fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 console.log('H55_REAL_EDGE_RAW_ACTION_REPLAY',JSON.stringify(langs),'ORIGIN',attempts.origin,
             'MODEL_SUCCESS_QUALIFIED',false);
})().catch(e=>{console.error(e.stack);process.exitCode=1});