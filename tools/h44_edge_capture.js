/* H44 actual offscreen Edge Chromium capture; create-only and no internet.
 * Requires exactly Puppeteer-Core 25.12.0 from a controlled local package.
 * No privileged oracle/task answers are available in this browser process.
 */
'use strict';
const fs=require('fs');
const path=require('path');
const crypto=require('crypto');
const {pathToFileURL}=require('url');
const modulePath=process.env.PUPPETEER_CORE_PATH;
if(!modulePath||!fs.existsSync(path.join(modulePath,'package.json')))throw Error('Pinned local Puppeteer-Core path required');
const version=JSON.parse(fs.readFileSync(path.join(modulePath,'package.json'),'utf8')).version;
if(version!=='25.12.0')throw Error('Unexpected Puppeteer-Core '+version);
const puppeteer=require(modulePath);
const root=path.resolve(process.argv[2]||'h44/assets');
const scenes=JSON.parse(fs.readFileSync(path.join(root,'scenes.json'),'utf8'));
if(scenes.schema!=='hala.h44.original-html-plan.v1'||scenes.scenes.length!==24)throw Error('Wrong frozen source plan');
const src=path.join(root,'pages'),out=path.join(root,'screenshots');
if(fs.readdirSync(out).length)throw Error('Refuse overwrite original screenshot outputs');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Microsoft Edge binary unavailable');

(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,
   args:['--no-first-run','--disable-extensions','--disable-background-networking',
         '--disable-sync','--disable-component-update']});
 const record=[];
 try{
  const browserVersion=await browser.version();
  const page=await browser.newPage();
  await page.setViewport({width:960,height:540,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',req=>req.url().startsWith('file:')?req.continue():req.abort());
  for(const scene of scenes.scenes){
   const pageFile=path.join(src,scene.id+'.html');
   const html=fs.readFileSync(pageFile);
   if(sha(html)!==scene.page_sha256)throw Error('Source HTML changed '+scene.id);
   await page.goto(pathToFileURL(pageFile).href,{waitUntil:'load',timeout:10000});
   const dom=await page.evaluate(()=>{
    const elements=[...document.querySelectorAll('[data-ref]')];
    return {dir:document.documentElement.dir,language:document.documentElement.lang,
      controls:elements.map(e=>{let r=e.getBoundingClientRect();return {
        ref:e.dataset.ref,name:e.getAttribute('aria-label'),role:e.tagName==='INPUT'?'Edit':'Button',
        visible:!!(r.width>0&&r.height>0),bounds:[r.x,r.y,r.width,r.height]};})};
   });
   if(dom.controls.length!==5||dom.language!==scene.language||
      dom.dir!==(scene.rtl?'rtl':'ltr'))throw Error('DOM/a11y mismatch '+scene.id);
   const dest=path.join(out,scene.id+'.png');
   if(fs.existsSync(dest))throw Error('No screenshot overwrite '+scene.id);
   const png=Buffer.from(await page.screenshot({type:'png',fullPage:false,
      animations:'disabled',caret:'hide'}));
   if(png.subarray(0,8).toString('hex')!=='89504e470d0a1a0a'
      ||png.readUInt32BE(16)!==960||png.readUInt32BE(20)!==540)throw Error('PNG dimension error');
   fs.writeFileSync(dest,png,{flag:'wx'});
   record.push({id:scene.id,language:scene.language,layout:scene.layout,
      split:scene.split,family:scene.family,
      image_sha256:sha(png),image_bytes:png.length,
      page_sha256:sha(html),controls:dom.controls});
  }
  const metadata={schema:'hala.h44.edge-browser-capture.v1',
    playwright_core:null,puppeteer_core:version,
    browser_version:browserVersion,viewport:{width:960,height:540,device_scale_factor:1},
    original_h43_manifest_sha256:scenes.h43_manifest_sha256,
    scene_plan_sha256:sha(fs.readFileSync(path.join(root,'scenes.json'))),
    no_external_network:true,model_inference:false,native_actions:false,
    model_oracle_exposure:false,rendered_records:record};
  fs.writeFileSync(path.join(root,'capture.json'),JSON.stringify(metadata,null,2)+'\n',{flag:'wx'});
  console.log('H44_ACTUAL_CHROMIUM_SCREENSHOTS',record.length,'BROWSER',browserVersion);
  console.log('H44_CAPTURE_SHA256',sha(fs.readFileSync(path.join(root,'capture.json'))));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
