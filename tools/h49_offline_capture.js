/* H49: actual headless Edge screenshots of original offline fictional apps only.
 * No real Najiz, Absher, FHIR server, network, logins or native OS actuation.
 * Browser-only click replay checks the target and a wrong button separately.
 */
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const {pathToFileURL} = require('url');
const core = process.env.PUPPETEER_CORE_PATH;
if (!core || JSON.parse(fs.readFileSync(path.join(core,'package.json'),'utf8')).version !== '25.12.0')
    throw Error('Pinned Puppeteer Core 25.12.0 is required');
const puppeteer = require(core);
const root = path.resolve(process.argv[2] || 'h49/assets');
const sourceBytes = fs.readFileSync(path.join(root,'source.json'));
const source = JSON.parse(sourceBytes);
if (source.schema !== 'hala.h49.original-offline-domain-grounding.v1' ||
    source.count !== 96 || source.rows.length !== 96 || source.real_service_test !== false)
    throw Error('Invalid original H49 source');
const output = path.join(root,'png');
if (fs.readdirSync(output).length !== 0 || fs.existsSync(path.join(root,'capture.json')))
    throw Error('Create-only output: existing H49 capture');
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if (!fs.existsSync(edge)) throw Error('Microsoft Edge binary not found');
(async()=>{
 const browser = await puppeteer.launch({executablePath:edge,headless:true,
  args:['--no-first-run','--disable-extensions','--disable-background-networking',
        '--disable-sync','--disable-component-update']});
 try {
  const page=await browser.newPage();
  await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  let rejected=0, permitted=0;
  await page.setRequestInterception(true);
  page.on('request', req => {
   if(req.url().startsWith('file:')) {permitted++;req.continue();}
   else {rejected++;req.abort();}
  });
  const records=[];
  for(const row of source.rows) {
   const html=path.join(root,'html',row.id+'.html');
   const original=fs.readFileSync(html);
   if(hash(original)!==row.html_sha256)throw Error('HTML source differs: '+row.id);
   const url=pathToFileURL(html).href;
   await page.goto(url,{waitUntil:'load',timeout:20000});
   const observed=await page.evaluate(()=>({
    language:document.documentElement.lang,
    direction:document.documentElement.dir,
    controls:[...document.querySelectorAll('button.pick')].map(b=>{
      const r=b.getBoundingClientRect();
      return {index:Number(b.dataset.idx),key:Number(b.dataset.key),
              label:b.getAttribute('aria-label'),text:b.innerText,
              bounds:[r.x,r.y,r.width,r.height],visible:r.width>0&&r.height>0};
    })
   }));
   if(observed.controls.length!==6 || observed.language!==row.language ||
      observed.direction!==(row.language==='ar'?'rtl':'ltr') ||
      observed.controls.some(c=>!c.visible || c.bounds[0]<0 || c.bounds[1]<0 ||
       c.bounds[0]+c.bounds[2]>1024 || c.bounds[1]+c.bounds[3]>610))
      throw Error('Offscreen or broken original localized controls '+row.id);
   const png=Buffer.from(await page.screenshot({type:'png',animations:'disabled',caret:'hide'}));
   if(png.readUInt32BE(16)!==1024 || png.readUInt32BE(20)!==640)
       throw Error('Wrong original screenshot dimensions');
   fs.writeFileSync(path.join(output,row.id+'.png'),png,{flag:'wx'});
   const target=observed.controls[row.target_index];
   if(!target || target.key!==row.target_key)throw Error('Target-key mismatch');
   await page.locator('button.pick[data-idx="'+row.target_index+'"]').click();
   const successful=await page.evaluate(()=>({
       selected:Number(document.body.dataset.selected),
       clicks:Number(document.body.dataset.selectionCount)}));
   if(successful.selected!==row.target_key||successful.clicks!==1)
       throw Error('True postcondition mismatch on golden target '+row.id);
   await page.goto(url,{waitUntil:'load',timeout:20000});
   const wrong=(row.target_index+1)%6;
   await page.locator('button.pick[data-idx="'+wrong+'"]').click();
   const miss=await page.evaluate(()=>({
       selected:Number(document.body.dataset.selected),
       clicks:Number(document.body.dataset.selectionCount)}));
   if(miss.selected===row.target_key||miss.clicks!==1)
       throw Error('Incorrectly accepted realistic wrong target '+row.id);
   records.push({id:row.id,html_sha256:row.html_sha256,
       png_sha256:hash(png),png_bytes:png.length,split:row.split,
       family:row.family,language:row.language,controls:observed.controls,
       golden:{selected:successful.selected,clicks:successful.clicks},
       negative:{selected:miss.selected,clicks:miss.clicks,wrong_index:wrong}});
  }
  const capture={schema:'hala.h49.actual-offline-edge-capture.v1',
    source_sha256:hash(sourceBytes),browser_version:await browser.version(),
    puppeteer_core:'25.12.0',screenshots:records.length,width:1024,height:640,
    real_service_access:false,network_used:false,native_actions:false,
    model_inference:false,external_requests_rejected:rejected,
    file_requests_permitted:permitted,records};
  fs.writeFileSync(path.join(root,'capture.json'),JSON.stringify(capture,null,2)+'\n',{flag:'wx'});
  console.log('H49_ORIGINAL_EDGE_SCREENSHOTS',records.length,
    'REAL_SERVICE_ACCESS',capture.real_service_access,
    'SOURCE_SHA256',capture.source_sha256);
  console.log('H49_CAPTURE_SHA256',hash(fs.readFileSync(path.join(root,'capture.json'))));
 } finally {await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
