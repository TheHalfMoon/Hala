/* H55: REAL Edge-rendered original 24 bilingual drag DEV scenarios.
 * First-party only, no live service, no competitor source, no model.
 * PUBLIC DEV source for iteration: NEVER call it independent blind TEST.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{pathToFileURL}=require('url');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const core=process.env.PUPPETEER_CORE_PATH;
if(!core||JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0')
  throw Error('Pinned puppeteer-core 25.12.0 is required');
const puppeteer=require(core);
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Real Edge is not installed');
const root=path.resolve(process.argv[2]||'');
if(!root || fs.existsSync(root))throw Error('New H55 public DEV directory must not exist');
const names=[
 ['ar_civic','ar','مركز الخدمات التجريبي','ضبط نموذج خيالي'],
 ['ar_records','ar','مكتب السجلات التجريبي','تحديث طلب افتراضي'],
 ['ar_logistics','ar','بوابة الشحن الافتراضية','تعديل مستوى الشحن'],
 ['en_fhir','en','Synthetic FHIR Lab','Resource review simulator'],
 ['en_design','en','Fictional Design Studio','Animation playhead sandbox'],
 ['en_inventory','en','Fictional Inventory Lab','Stock allocation simulator']];
const escapeHTML=x=>String(x).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
const goals=[27,38,49,61,74,86],colors=['#246b89','#6d54a1','#347f67','#984b50'];
const plans=[];
for(const [family,language,title,note] of names)
 for(let i=0;i<4;i++){
  const n=plans.length;
  plans.push({id:'h55-'+family+'-'+String(i).padStart(2,'0'),family,language,title,note,
    split:'dev',viewport:[1024,640],start_percent:10,
    goal_percent:goals[(n*5+i)%6],track_left:126+(n%4)*30,
    track_top:115+(n%3)*18,track_width:610-(n%4)*25,
    color:colors[n%4],background:['#f0f5fa','#f6f2ea','#eef7f2'][n%3]});
 }
const canonical=v=>Buffer.from(JSON.stringify(v,null,2)+'\n','utf8');
function render(p){
 const ar=p.language==='ar',goal=p.goal_percent,accent=p.color;
 const inst=ar?'اسحب المؤشر إلى '+goal+'٪. لا تستخدم النقر.':'Drag the slider handle to '+goal+'%. Do not click.';
 const notice=ar?'واجهة خيالية دون الاتصال بجهات حكومية':'Fictional original interface. No real accounts or services.';
 return Buffer.from([
 '<!doctype html><html lang="'+p.language+'" dir="'+(ar?'rtl':'ltr')+'"><head><meta charset="utf-8">',
 '<meta name="viewport" content="width=1024,initial-scale=1">',
 '<style>*{box-sizing:border-box}body{margin:0;background:'+p.background+';color:#253547;font:17px Tahoma,Segoe UI,Arial,sans-serif}',
 'header{height:76px;background:#fff;border-bottom:2px solid #cedde8;padding:20px 45px;display:flex;justify-content:space-between}',
 'main{height:564px;position:relative}h1{text-align:center;font-size:22px;margin:37px 0 10px}',
 '.instruction{text-align:center;font-weight:700;color:'+accent+'}',
 '.panel{height:318px;background:#fff;border:1px solid #c6d3e1;border-radius:14px;margin:22px 52px;position:relative}',
 '#track{direction:ltr;position:absolute;left:'+p.track_left+'px;top:'+p.track_top+'px;width:'+p.track_width+'px;height:22px;border-radius:14px;background:#d5e3eb;border:1px solid #89a6b8;touch-action:none}',
 '#fill{height:100%;width:10%;background:'+accent+';border-radius:14px;opacity:.55;pointer-events:none}',
 '#handle{position:absolute;top:-16px;left:10%;margin-left:-21px;width:42px;height:54px;background:'+accent+';border:5px solid #fff;border-radius:19px;box-shadow:0 2px 5px #777;touch-action:none;cursor:grab;z-index:5}',
 '#target{position:absolute;left:'+goal+'%;top:-35px;width:3px;height:80px;background:#b75235;pointer-events:none}',
 '#target::after{content:"'+goal+'%";position:absolute;left:-23px;top:-29px;color:#b75235;font:bold 18px Arial}',
 '.scale{position:absolute;top:62px;left:0;width:100%;display:flex;justify-content:space-between;font-size:13px;color:#526b83}',
 'footer{position:fixed;bottom:18px;text-align:center;width:100%;font-size:12px;color:#5a7086}</style></head>',
 '<body><header><strong>'+escapeHTML(p.title)+'</strong><small>H55 PUBLIC DEV — '+p.id+'</small></header>',
 '<main><h1>'+escapeHTML(p.note)+'</h1><div class="instruction">'+escapeHTML(inst)+'</div>',
 '<div class="panel"><div id="track"><div id="fill"></div><div id="target"></div><div id="handle" role="slider"></div>',
 '<div class="scale"><span>0%</span><span>20%</span><span>40%</span><span>60%</span><span>80%</span><span>100%</span></div>',
 '</div></div></main><footer>'+escapeHTML(notice)+'</footer>',
 '<script>',
 "const h=document.getElementById('handle'),t=document.getElementById('track'),f=document.getElementById('fill');",
 'const START=10,TARGET='+goal+';let active=false,pid=null,x0=0,moves=0;',
 "document.body.dataset.completed='0';document.body.dataset.rejected='0';document.body.dataset.accepted='false';",
 'function pct(x){const r=t.getBoundingClientRect();return Math.min(100,Math.max(0,(x-r.left)/r.width*100));}',
 "function setPct(n){h.style.left=n+'%';f.style.width=n+'%';}",
 "h.addEventListener('pointerdown',e=>{if(active||document.body.dataset.accepted==='true')return;active=true;pid=e.pointerId;x0=e.clientX;moves=0;h.setPointerCapture(e.pointerId);e.preventDefault();});",
 "h.addEventListener('pointermove',e=>{if(!active||e.pointerId!==pid)return;moves++;setPct(pct(e.clientX));});",
 "h.addEventListener('pointerup',e=>{if(!active||e.pointerId!==pid)return;active=false;const ok=moves>=2&&Math.abs(pct(e.clientX)-TARGET)<=1.4&&Math.abs(e.clientX-x0)>=40;if(ok){document.body.dataset.accepted='true';document.body.dataset.completed='1';setPct(TARGET);}else{document.body.dataset.rejected=String(Number(document.body.dataset.rejected)+1);setPct(START);}});",
 "h.addEventListener('pointercancel',()=>{active=false;setPct(START);});",
 '</script></body></html>'].join('\n'),'utf8');
}
(async()=>{
 fs.mkdirSync(root,{recursive:false});fs.mkdirSync(path.join(root,'html'));fs.mkdirSync(path.join(root,'png'));
 const rights={origin:'first_party_fictional_slider_public_dev',license:'CC0-1.0',
  external_images:false,live_services:false,train_eligible:false,official_benchmark:false};
 fs.writeFileSync(path.join(root,'plans.json'),canonical(plans),{flag:'wx'});
 const scenes=[];
 for(const p of plans){const b=render(p);fs.writeFileSync(path.join(root,'html',p.id+'.html'),b,{flag:'wx'});scenes.push({id:p.id,html_sha256:sha(b)});}
 const source={schema:'hala.h55.original-bilingual-drag-public-dev.v1',rights,
  split:'dev',public_dev_only:true,blind_test:false,trained_model_predictions:0,
  plans_sha256:sha(canonical(plans)),scenes};
 fs.writeFileSync(path.join(root,'source.json'),canonical(source),{flag:'wx'});
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
  '--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update']});
 let blocked=0;const cases=[];
 try{
  const page=await browser.newPage();await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',req=>{if(req.url().startsWith('file:'))req.continue();else{blocked++;req.abort();}});
  for(const p of plans){
   await page.goto(pathToFileURL(path.join(root,'html',p.id+'.html')).href,{waitUntil:'load'});
   const geometry=await page.evaluate(()=>{
     const a=(id)=>{const b=document.getElementById(id).getBoundingClientRect();return [b.x,b.y,b.width,b.height];};
     return {track:a('track'),handle:a('handle'),target:a('target'),body:document.body.dataset.accepted};
   });
   if(geometry.body!=='false')throw Error('Scene was already completed');
   const [tx,ty,tw,th]=geometry.track,[hx,hy,hw,hh]=geometry.handle;
   const start=[Math.round(hx+hw/2),Math.round(hy+hh/2)];
   const end=[Math.round(tx+tw*p.goal_percent/100),start[1]];
   if(start[0]<tx||end[0]>tx+tw||start[1]<80||start[1]>590||end[0]-start[0]<40)
     throw Error('Out-of-bounds original slider geometry '+p.id);
   const screenshot=await page.screenshot({type:'png',captureBeyondViewport:false});
   fs.writeFileSync(path.join(root,'png',p.id+'.png'),screenshot,{flag:'wx'});
   const state=()=>page.evaluate(()=>({accepted:document.body.dataset.accepted,
       completed:Number(document.body.dataset.completed),rejected:Number(document.body.dataset.rejected)}));
   const drag=async(a,b)=>{
     await page.mouse.move(...a);await page.mouse.down();await page.mouse.move(...b,{steps:12});await page.mouse.up();
   };
   // Intentionally wrong origin: clicking track away from the handle must not drag.
   await drag([start[0]+75,start[1]],end);
   const afterWrongOrigin=await state();
   if(afterWrongOrigin.accepted!=='false'||afterWrongOrigin.completed!==0)
     throw Error('Track background incorrectly accepted as slider handle '+p.id);
   // Real drag that ends at an incorrect target must be rejected by the browser.
   const badEnd=[Math.round(tx+tw*(p.goal_percent>55?20:90)/100),start[1]];
   await drag(start,badEnd);
   const afterWrongEnd=await state();
   if(afterWrongEnd.accepted!=='false'||afterWrongEnd.completed!==0||afterWrongEnd.rejected!==1)
     throw Error('Wrong actual drag accepted '+p.id+' '+JSON.stringify(afterWrongEnd));
   await drag(start,end);
   const afterCorrect=await state();
   if(afterCorrect.accepted!=='true'||afterCorrect.completed!==1||afterCorrect.rejected!==1)
     throw Error('True original drag did not complete '+p.id+' '+JSON.stringify(afterCorrect));
   const prompt=(p.language==='ar'?'اسحب مؤشر الواجهة الخيالية إلى ':'Drag the fictional interface slider to ')+
     p.goal_percent+'%. '+(p.language==='ar'?'أعد فقط فعل السحب بإحداثيات الصورة الأصلية.':'Return ONLY one native drag action with original-image pixel coordinates.')+
     ' Screenshot 1024x640.';
   cases.push({id:p.id,family:p.family,language:p.language,split:'dev',prompt,
     image_path:'png/'+p.id+'.png',image_sha256:sha(screenshot),source_html_sha256:scenes.find(x=>x.id===p.id).html_sha256,
     start_point:start,end_point:end,track_bbox:geometry.track,handle_bbox:geometry.handle,
     goal_percent:p.goal_percent,source_image_dimensions:[1024,640],
     wrong_origin_rejected:afterWrongOrigin.completed===0,wrong_endpoint_rejected:afterWrongEnd.rejected===1,
     correct_drag_completed:afterCorrect.completed===1,
     original_browser:'Microsoft Edge real pointerdown/move/up',model_inference:false});
   if(cases.length%6===0)console.log('REAL_EDGE_H55_CAPTURED',cases.length);
  }
 }finally{await browser.close();}
 if(cases.length!==24||new Set(cases.map(x=>x.image_sha256)).size!==24||
    cases.filter(x=>x.language==='ar').length!==12||cases.filter(x=>x.language==='en').length!==12)
   throw Error('Insufficient unique original bilingual screenshots');
 const receipt={schema:'hala.h55.real-edge-native-drag-public-dev-replay.v1',
  source_sha256:sha(canonical(source)),scenes:24,arabic:12,english:12,
  real_browser_correct_drags:24,real_browser_wrong_endpoint_rejections:24,
  real_browser_wrong_origin_rejections:24,source_train:0,
  real_model_inference:false,blind_benchmark_test:false,external_requests_blocked:blocked,cases};
 fs.writeFileSync(path.join(root,'capture.json'),canonical(receipt),{flag:'wx'});
 console.log('H55_BROWSER_REAL_DRAG_DEV_24_PASS','SOURCE_SHA256',receipt.source_sha256);
})().catch(e=>{console.error(e.stack);process.exitCode=1});