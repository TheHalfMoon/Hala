/* H48: real Edge headless Chromium pixels from 96 original fictional pages.
   No network, user app, privileged native click, or model inference. */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const pkg=process.env.PUPPETEER_CORE_PATH;
if(!pkg||!fs.existsSync(path.join(pkg,'package.json')))throw Error('Pinned local puppeteer-core missing');
if(JSON.parse(fs.readFileSync(path.join(pkg,'package.json'),'utf8')).version!=='25.12.0')
 throw Error('Unexpected Puppeteer-Core version');
const puppeteer=require(pkg);
const root=path.resolve(process.argv[2]||'h48/assets');
const original=fs.readFileSync(path.join(root,'source_manifest.json'));
const source=JSON.parse(original);
if(source.schema!=='hala.h48.diverse-first-party-visual-grounding.v1' ||
 source.count!==96 ||source.rows.length!==96)throw Error('H48 original source wrong');
const out=path.join(root,'png');
if(fs.readdirSync(out).length ||fs.existsSync(path.join(root,'capture.json')))throw Error('No H48 overwrite');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('No installed Edge found');
(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,
  args:['--no-first-run','--disable-extensions','--disable-background-networking',
        '--disable-sync','--disable-component-update']});
 try{
  const page=await browser.newPage();
  await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',request=>request.url().startsWith('file:')?request.continue():request.abort());
  const observations=[];
  for(const scene of source.rows){
    const htmlPath=path.join(root,'html',scene.id+'.html');
    const payload=fs.readFileSync(htmlPath);
    if(hash(payload)!==scene.html_sha256)throw Error('Original HTML mismatch '+scene.id);
    await page.goto(pathToFileURL(htmlPath).href,{waitUntil:'load',timeout:15000});
    const observed=await page.evaluate(()=>{
      return {lang:document.documentElement.lang,dir:document.documentElement.dir,
        controls:[...document.querySelectorAll('button.act')].map(b=>{
          const r=b.getBoundingClientRect();
          return {index:Number(b.dataset.idx),name:b.getAttribute('aria-label'),
              text:b.textContent.trim(),bounds:[r.x,r.y,r.width,r.height],
              visible:r.width>0&&r.height>0}})};
    });
    if(observed.controls.length!==6||observed.lang!==scene.language ||
       observed.dir!==(scene.language==='ar'?'rtl':'ltr'))throw Error('Incomplete localized GUI '+scene.id);
    const png=Buffer.from(await page.screenshot({type:'png',fullPage:false,
          animations:'disabled',caret:'hide'}));
    if(png.subarray(0,8).toString('hex')!=='89504e470d0a1a0a' ||
       png.readUInt32BE(16)!==1024||png.readUInt32BE(20)!==640)throw Error('Invalid screenshot '+scene.id);
    fs.writeFileSync(path.join(out,scene.id+'.png'),png,{flag:'wx'});
    observations.push({id:scene.id,source_sha256:scene.html_sha256,
       png_sha256:hash(png),png_bytes:png.length,language:scene.language,
       split:scene.split,family:scene.family,controls:observed.controls});
  }
  const capture={schema:'hala.h48.original-edge-pixels.v1',
    browser_version:await browser.version(),puppeteer_core:'25.12.0',
    width:1024,height:640,source_manifest_sha256:hash(original),
    screenshots:observations.length,network_used:false,native_actions:false,
    model_inference:false,records:observations};
  fs.writeFileSync(path.join(root,'capture.json'),JSON.stringify(capture,null,2)+'\n',{flag:'wx'});
  console.log('H48_GENUINE_OFFLINE_EDGE_PNGS',observations.length,'SOURCE',capture.source_manifest_sha256);
  console.log('H48_CAPTURE_SHA256',hash(fs.readFileSync(path.join(root,'capture.json'))));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1});
