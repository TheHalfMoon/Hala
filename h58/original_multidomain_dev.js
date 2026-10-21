/* H58 real Microsoft Edge native drag PUBLIC DEV, 24 entirely original mock UIs.
 * Source-author actions are only harness qualification, never neural predictions.
 */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{pathToFileURL}=require('url');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const canonical=x=>Buffer.from(JSON.stringify(x,null,2)+'\n','utf8');
const core=process.env.PUPPETEER_CORE_PATH;
if(!core||JSON.parse(fs.readFileSync(path.join(core,'package.json'))).version!=='25.12.0')throw Error('Pinned Puppeteer-core v25.12.0 required');
const puppeteer=require(core),edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
if(!fs.existsSync(edge))throw Error('Installed genuine Edge required');
if(!process.argv[2]||fs.existsSync(process.argv[2]))throw Error('Create-only original DEV root required');
const root=path.resolve(process.argv[2]);
const kinds=['sheet','slide_resize','text_highlight'],entries=[];
for(const kind of kinds)for(let i=0;i<8;i++){
 const ar=(i%2===0),x0=185+(i%3)*35,y0=275+(i%4)*24;
 const x1=x0+(kind==='slide_resize'?160:255)+(i%2)*40;
 const y1=kind==='sheet'?y0+42:kind==='text_highlight'?y0+17:y0;
 const id='h58-'+kind+'-'+String(i).padStart(2,'0');
 entries.push({id,kind,language:ar?'ar':'en',ordered:kind==='slide_resize',
   split:'dev',image_size:[1024,640],family:'fictional_firstparty_'+kind,
   title:ar?{sheet:'جدول المخزون الافتراضي',slide_resize:'استوديو الشرائح الافتراضي',text_highlight:'ملاحظات البحث الافتراضية'}[kind]:
     {sheet:'Original Inventory Sheet',slide_resize:'Original Slide Editor',text_highlight:'Original Research Notes'}[kind],
   instruction:ar?{sheet:'حدد الخلايا من البداية إلى النهاية',slide_resize:'اسحب مقبض الحجم إلى الهدف',text_highlight:'ظلل الكلمات من البداية إلى النهاية'}[kind]:
     {sheet:'Select the cells from START to END',slide_resize:'Drag the right size handle to TARGET',text_highlight:'Highlight from START to END'}[kind],
   start:[x0,y0],end:[x1,y1],bad_start:[x0,y0+96],bad_end:[x1,y1+95],
   color:['#226c84','#765c94','#3c856c','#ad6a50'][i%4]});
}
const escapeHTML=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
function render(p){
 const x0=p.start[0],y0=p.start[1],x1=p.end[0],y1=p.end[1];
 const ar=p.language==='ar',sheet=p.kind==='sheet',slide=p.kind==='slide_resize';
 const behind=sheet?
   '<div class="scene" style="top:154px;background:#eaf1f7"><b>Row &nbsp;&nbsp; Product &nbsp;&nbsp; Date &nbsp;&nbsp; Quantity</b><hr>'+
   Array.from({length:7},(_,j)=>'<p>ROW-'+(300+j)+' &nbsp;&nbsp; ITEM &nbsp;&nbsp; 2026-10 &nbsp;&nbsp; '+(12+j*4)+'</p>').join('')+'</div>':
   slide?
   '<div class="scene" style="top:153px;background:#e6ecfb;border:3px solid '+p.color+'"><b>Hala Slide</b><p>Fictional original editor</p><p>Drag a resize handle</p></div>':
   '<div class="scene" style="top:153px"><b>Research summary</b><p>These entirely fictional words are available for selection.</p><p>Research notes and document editing demo.</p></div>';
 const point=(xy,id,label,background)=>'<div class="drag-point" id="'+id+'" style="left:'+(xy[0]-25)+'px;top:'+(xy[1]-19)+'px;background:'+background+'">'+escapeHTML(label)+'</div>';
 const from=ar?'بدء':'START',to=ar?'نهاية':'END';
 return Buffer.from([
 '<!doctype html><html lang="'+p.language+'" dir="ltr"><head><meta charset="utf-8">',
 '<meta name="viewport" content="width=1024,initial-scale=1">',
 '<style>*{box-sizing:border-box}body{margin:0;background:#eff5f8;color:#253b4b;font:17px Segoe UI,Tahoma,Arial,sans-serif}',
 'header{height:74px;background:#fff;padding:18px 35px;border-bottom:2px solid #d2dfe8}main{position:relative;width:1024px;height:566px}',
 'h1{margin:19px 50px;font-size:24px}.instruction{margin:10px 50px;color:'+p.color+';font-size:21px;font-weight:700}',
 '.scene{position:absolute;left:120px;width:780px;height:340px;border-radius:14px;padding:25px;top:155px;color:#5e7485;font-size:16px;background:#fff;border:1px solid #cedce5}',
 '.scene p{border-bottom:1px solid #e0e9ed;padding-bottom:4px}.drag-point{z-index:5;position:absolute;width:65px;height:43px;display:flex;align-items:center;justify-content:center;',
 'color:#fff;border:3px solid #fff;border-radius:9px;font-weight:700;font-size:12px;cursor:crosshair;touch-action:none;user-select:none;box-shadow:0 2px 6px #879fae}',
 'footer{position:absolute;bottom:22px;left:42px;font-size:13px;color:#7890a0}</style></head>',
 '<body><header><strong>'+escapeHTML(p.title)+'</strong><small style="float:right">H58 FIRST-PARTY PUBLIC DEV</small></header>',
 '<main><h1>'+escapeHTML(p.kind.replace("_"," "))+'</h1><div class="instruction">'+escapeHTML(p.instruction)+'</div>',
 behind,point(p.start,'start',slide?'↔':from,p.color),point(p.end,'end',slide?'TARGET':to,'#ac5839'),
 '<footer>No real users, services, external application screenshots or Hcompany benchmark labels</footer></main>',
 '<script>(()=>{const A='+JSON.stringify(p.start)+',B='+JSON.stringify(p.end)+',ORDERED='+JSON.stringify(p.ordered)+';',
 'let start=null,active=false,moves=0;const near=(x,y)=>Math.hypot(x[0]-y[0],x[1]-y[1])<22;',
 'document.body.dataset.completed="0";document.body.dataset.accepted="false";document.body.dataset.rejected="0";',
 'document.addEventListener("pointerdown",e=>{start=[e.clientX,e.clientY];active=true;moves=0;e.preventDefault();});',
 'document.addEventListener("pointermove",()=>{if(active)moves++;});',
 'document.addEventListener("pointerup",e=>{if(!active)return;active=false;const end=[e.clientX,e.clientY];',
 'const good=moves>=3&&((near(start,A)&&near(end,B))||(!ORDERED&&near(start,B)&&near(end,A)));',
 'if(good){document.body.dataset.completed="1";document.body.dataset.accepted="true";}',
 'else document.body.dataset.rejected=String(Number(document.body.dataset.rejected)+1);});',
 'document.addEventListener("pointercancel",()=>{active=false;});})();</script></body></html>'
 ].join('\n'),'utf8');
}
(async()=>{
 fs.mkdirSync(root);fs.mkdirSync(path.join(root,'html'));fs.mkdirSync(path.join(root,'png'));
 fs.writeFileSync(path.join(root,'plans.json'),canonical(entries),{flag:'wx'});
 const refs=[];
 for(const p of entries){const b=render(p);fs.writeFileSync(path.join(root,'html',p.id+'.html'),b,{flag:'wx'});refs.push({id:p.id,html_sha256:sha(b)});}
 const source={schema:'hala.h58.original-multidomain-public-dev.v1',split:'dev',source:'first-party-original-fictional',
  rights:{license:'CC0-1.0',external_images:false,live_services:false,training_allowed:false,official_benchmark:false},
  public_dev_only:true,blind_test:false,trained_models_evaluated:0,plans_sha256:sha(canonical(entries)),scenes:refs};
 fs.writeFileSync(path.join(root,'source.json'),canonical(source),{flag:'wx'});
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
   '--no-first-run','--disable-extensions','--disable-background-networking','--disable-sync','--disable-component-update']});
 let blocked=0;const cases=[];
 try{
  const page=await browser.newPage();await page.setViewport({width:1024,height:640,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',r=>{if(r.url().startsWith('file:'))r.continue();else{blocked++;r.abort();}});
  const perform=async(a,b)=>{
    await page.mouse.move(...a);await page.mouse.down();await page.mouse.move(...b,{steps:12});await page.mouse.up();
    return await page.evaluate(()=>({accepted:document.body.dataset.accepted==='true',
      complete:Number(document.body.dataset.completed),rejected:Number(document.body.dataset.rejected)}));
  };
  for(const p of entries){
   const url=pathToFileURL(path.join(root,'html',p.id+'.html')).href;
   const reload=()=>page.goto(url,{waitUntil:'load'});
   await reload();const png=await page.screenshot({type:'png',captureBeyondViewport:false});
   if(png.readUInt32BE(16)!==1024||png.readUInt32BE(20)!==640)throw Error('Incorrect Edge screenshot dimensions');
   fs.writeFileSync(path.join(root,'png',p.id+'.png'),png,{flag:'wx'});
   const wrongStart=await perform(p.bad_start,p.end);if(wrongStart.complete!==0)throw Error('Wrong-origin accepted '+p.id);
   await reload();const wrongEnd=await perform(p.start,p.bad_end);if(wrongEnd.complete!==0)throw Error('Wrong-end accepted '+p.id);
   await reload();const correct=await perform(p.start,p.end);if(correct.complete!==1)throw Error('Original correct drag failed '+p.id);
   await reload();const reverse=await perform(p.end,p.start);if(reverse.complete!==(p.ordered?0:1))throw Error('Reverse ordered semantics failed '+p.id);
   cases.push({id:p.id,domain:p.kind,language:p.language,ordered:p.ordered,
     image_path:'png/'+p.id+'.png',image_sha256:sha(png),source_html_sha256:refs.find(x=>x.id===p.id).html_sha256,
     image_size:p.image_size,split:'dev',start_point:p.start,end_point:p.end,
     bad_start_rejected:wrongStart.complete===0,bad_end_rejected:wrongEnd.complete===0,
     original_forward_completed:correct.complete===1,original_reverse_completed:reverse.complete===1});
   if(cases.length%8===0)console.log('H58_GENUINE_ORIGINAL_EDGE_CAPTURE',cases.length);
  }
 }finally{await browser.close();}
 if(cases.length!==24||new Set(cases.map(c=>c.image_sha256)).size!==24||
    cases.filter(c=>c.language==='ar').length!==12||cases.filter(c=>c.language==='en').length!==12)
   throw Error('Bad unique bilingual source DEV inventory');
 const receipt={schema:'hala.h58.real-edge-multidomain-public-dev.v1',
    original_source_sha256:sha(canonical(source)),scenes:24,domains:{sheet:8,slide_resize:8,text_highlight:8},
    arabic:12,english:12,correct_author_oracle_drags:24,
    wrong_start_rejected:24,wrong_end_rejected:24,
    unordered_reverse_completed:16,ordered_reverse_rejected:8,
    network_requests_blocked:blocked,actual_model_inference:false,
    public_dev_only:true,blind_test:false,train_rows:0,cases};
 fs.writeFileSync(path.join(root,'capture.json'),canonical(receipt),{flag:'wx'});
 console.log('H58_GENUINE_EDGE_MIXED_DRAG_HARNESS_READY',cases.length,receipt.original_source_sha256);
})().catch(e=>{console.error(e.stack);process.exitCode=1});