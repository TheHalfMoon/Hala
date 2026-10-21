/* H58 replay UNALTERED future raw agent native-drag JSON inside genuine Edge.
 * Original source oracle answers are reference ONLY, never substituted for model proposals.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{pathToFileURL}=require('url');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const core=process.env.PUPPETEER_CORE_PATH,edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!core||JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0'||!fs.existsSync(edge))
  throw Error('Real installed Edge + pinned Puppeteer-core 25.12.0 required');
const puppeteer=require(core);
const root=path.resolve(process.argv[2]||''),
  input=path.resolve(process.argv[3]||''),out=path.resolve(process.argv[4]||'');
if(!process.argv[2]||!process.argv[3]||!process.argv[4]||!fs.existsSync(root)||!fs.existsSync(input)||fs.existsSync(out))
  throw Error('Frozen source+raw candidates and CREATE-ONLY output required');
const source=JSON.parse(fs.readFileSync(path.join(root,'source.json'),'utf8'));
const receipt=JSON.parse(fs.readFileSync(path.join(root,'capture.json'),'utf8'));
const candidates=JSON.parse(fs.readFileSync(input,'utf8'));
const expectedSHA='a1c17c5dca028096bdd653de927fae18b1797a241ca3529646e10a1eb5f4f3de';
if(sha(fs.readFileSync(path.join(root,'source.json')))!==expectedSHA ||
    receipt.original_source_sha256!==expectedSHA ||
    candidates.schema!=='hala.h58.raw-public-dev-native-drag-attempts.v1'||
    candidates.source_sha256!==expectedSHA||
    !['synthetic_unit_fixture','provider_raw_model_outputs'].includes(candidates.origin)||
    !Array.isArray(candidates.cases)||candidates.cases.length!==24)
  throw Error('H58 original 24 source-bound external raw predictions only');
const originals=new Map(receipt.cases.map(x=>[x.id,x]));
const map=new Map();
for(const c of candidates.cases){
 if(!c||Object.keys(c).sort().join()!=='id,image_sha256,raw_output'||
    !originals.has(c.id)||map.has(c.id)||c.image_sha256!==originals.get(c.id).image_sha256||
    typeof c.raw_output!=='string'||c.raw_output.length>4096)
   throw Error('Missing/source-mismatched raw agent response');
 map.set(c.id,c.raw_output);
}
function parse(raw){
 let j;try{j=JSON.parse(raw)}catch{return null}
 if(!j||Array.isArray(j)||Object.keys(j).sort().join()!=='coordinate_space,end_point,ordered,start_point,type'||
    j.type!=='drag'||j.coordinate_space!=='source_image_pixels'||typeof j.ordered!=='boolean')return null;
 for(const k of ['start_point','end_point']){
  let p=j[k];if(!Array.isArray(p)||p.length!==2||
    !p.every((v,i)=>Number.isInteger(v)&&v>=0&&v<(i===0?1024:640)))return null;
 }
 return j;
}
(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
  '--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update']});
 let rejectedNetwork=0;const results=[];
 try{
  const page=await browser.newPage();
  await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',r=>{if(r.url().startsWith('file:'))r.continue();else{rejectedNetwork++;r.abort();}});
  const verifyHTML=new Map(source.scenes.map(s=>[s.id,s.html_sha256]));
  for(const c of receipt.cases){
   const pageFile=path.join(root,'html',c.id+'.html'),pngFile=path.join(root,c.image_path);
   if(sha(fs.readFileSync(pageFile))!==c.source_html_sha256||
      sha(fs.readFileSync(pageFile))!==verifyHTML.get(c.id)||
      sha(fs.readFileSync(pngFile))!==c.image_sha256)
      throw Error('Original source scene changed since authored Edge evidence');
   await page.goto(pathToFileURL(pageFile).href,{waitUntil:'load'});
   const originalRaw=map.get(c.id),proposal=parse(originalRaw);
   // A syntactically valid action with the WRONG ordered semantics is a rejected action,
   // not an accepted success simply because the DOM pointer coordinates happen to work.
   const admissible=proposal&&proposal.ordered===c.ordered;
   if(admissible){
    await page.mouse.move(...proposal.start_point);await page.mouse.down();
    await page.mouse.move(...proposal.end_point,{steps:12});await page.mouse.up();
   }
   const state=await page.evaluate(()=>({accepted:document.body.dataset.accepted==='true',
     complete:Number(document.body.dataset.completed),rejected:Number(document.body.dataset.rejected)}));
   results.push({id:c.id,domain:c.domain,language:c.language,ordered:c.ordered,
      raw_valid:!!proposal,ordered_semantics_valid:!!admissible,
      genuine_edge_gui_completed:!!admissible&&state.complete===1&&state.accepted,
      observed_state:state});
  }
 }finally{await browser.close()}
 const by={};for(const c of results){
  const k=c.domain+'/'+c.language,b=by[k]??(by[k]={total:0,valid:0,real_edge_gui_success:0});
  b.total++;b.valid+=Number(c.raw_valid&&c.ordered_semantics_valid);
  b.real_edge_gui_success+=Number(c.genuine_edge_gui_completed);
 }
 const report={schema:'hala.h58.actual-edge-raw-drag-replay.v1',
  source_sha256:expectedSHA,origin:candidates.origin,candidate_input_sha256:sha(fs.readFileSync(input)),
  real_microsoft_edge_pointer_actions_executed:true,
  synthetic_fixture_only:candidates.origin==='synthetic_unit_fixture',
  raw_external_model_provenance_independently_verified:false,
  public_dev_only:true,blind_test:false,training_performed:false,
  official_benchmark_competitor_claim_qualified:false,external_requests_rejected:rejectedNetwork,
  by_domain_language:by,real_browser_success_total:results.filter(r=>r.genuine_edge_gui_completed).length,
  results};
 fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
 console.log('H58_ACTUAL_EDGE_PUBLIC_DEV',JSON.stringify(by),'SYNTHETIC_FIXTURE_ONLY',report.synthetic_fixture_only);
})().catch(e=>{console.error(e.stack);process.exitCode=1});