/* Actual original offline browser verification of all six H49 TRAIN options.
 * This proves local mock DOM click postconditions, NOT model accuracy.
 * Every request other than file:// is blocked. Create-only output.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const core=process.env.PUPPETEER_CORE_PATH;
if(!core || JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0')
  throw Error('Pinned puppeteer-core 25.12.0 required');
const puppeteer=require(core);
const root=path.resolve(process.argv[2]||'.');
const h49=path.join(root,'h49','assets'),h50=path.join(root,'h50','assets');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const manifest=JSON.parse(fs.readFileSync(path.join(h49,'sft_manifest.json')));
const provenance=JSON.parse(fs.readFileSync(path.join(h50,'source_provenance.json')));
const output=path.join(h50,'click_replay.json');
if(fs.existsSync(output))throw Error('Create-only: existing replay');
if(provenance.cases.length!==64 || manifest.source_sha256!==provenance.source_sha256 ||
    manifest.capture_sha256!==provenance.capture_sha256)throw Error('Original source identity mismatch');
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Edge missing');
(async()=>{
 const browser=await puppeteer.launch({headless:true,executablePath:edge,
   args:['--no-first-run','--disable-extensions','--disable-background-networking',
         '--disable-sync','--disable-component-update']});
 try{
   const page=await browser.newPage();
   await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
   await page.setRequestInterception(true);
   page.on('request',req=>req.url().startsWith('file:')?req.continue():req.abort());
   const cases=[];
   for(const original of provenance.cases){
     const html=path.join(h49,'html',original.source_id+'.html');
     const bytes=fs.readFileSync(html);
     if(sha(bytes)!==original.html_sha256 ||
       sha(fs.readFileSync(path.join(h49,'png',original.source_id+'.png')))!==original.png_sha256)
       throw Error('Original HTML/PNG digest changed: '+original.source_id);
     await page.goto(pathToFileURL(html).href,{waitUntil:'load',timeout:20000});
     const controls=await page.evaluate(()=>[...document.querySelectorAll('button.pick')].map(b=>{
        const r=b.getBoundingClientRect();
        return {key:Number(b.dataset.key),visible:r.width>50&&r.height>=40&&r.x>=0&&r.y>=0&&
          r.right<=1024&&r.bottom<=610};
     }));
     if(controls.length!==6 || controls.some(c=>!c.visible) ||
       new Set(controls.map(c=>c.key)).size!==6)throw Error('Offscreen/ambiguous controls: '+original.source_id);
     const selected=[],counts=[];
     for(let key=0;key<6;key++){
       await page.locator('button.pick[data-key="'+key+'"]').click();
       const state=await page.evaluate(()=>({
           selected:Number(document.body.dataset.selected),
           count:Number(document.body.dataset.selectionCount)}));
       if(state.selected!==key || state.count!==key+1)throw Error('True postcondition mismatch '+original.source_id+' '+key);
       selected.push(state.selected);counts.push(state.count);
     }
     cases.push({...original,clicked_keys:[0,1,2,3,4,5],
                 selected_keys:selected,click_counts:counts});
   }
   const receipt={schema:'hala.h50.original-browser-six-clicks.v1',
      source_sha256:manifest.source_sha256,capture_sha256:manifest.capture_sha256,
      case_count:64,click_count:384,real_service_access:false,
      model_inference:false,cases};
   fs.writeFileSync(output,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
   console.log('H50_BROWSER_CLICK_POSTCONDITIONS',cases.length*6,
      'GITHUB_EVIDENCE_SHA256',sha(fs.readFileSync(output)),
      'REAL_SERVICE_ACCESS',receipt.real_service_access);
 }finally{await browser.close();}
})().catch(err=>{console.error(err.stack);process.exitCode=1});
