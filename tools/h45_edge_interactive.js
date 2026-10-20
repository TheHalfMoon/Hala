/* H45 original first-party interactive HTML app browser actions only.
   No external navigation, no real user account/desktop UI, no model inference.
   Input oracle actions originate from H43 ONLY in this private evaluator.
*/
"use strict";
const fs=require("fs"),path=require("path"),crypto=require("crypto");
const {pathToFileURL}=require("url");
const lib=process.env.PUPPETEER_CORE_PATH;
if(!lib||!fs.existsSync(path.join(lib,"package.json")))throw Error("Explicit local PUPPETEER_CORE_PATH required");
const ver=JSON.parse(fs.readFileSync(path.join(lib,"package.json"),"utf8")).version;
if(ver!=="25.12.0")throw Error("Pinned puppeteer-core version mismatch");
const puppeteer=require(lib);
const root=path.resolve(process.argv[2]||"h45/assets");
const orig=JSON.parse(fs.readFileSync(path.resolve("h43/data/pilot_manifest.json"),"utf8"));
const plan=JSON.parse(fs.readFileSync(path.join(root,"scenes.json"),"utf8"));
if(orig.cases.length!==24 || plan.schema!=="hala.h45.original-browser-workflows.v1"
   ||plan.scenes.length!==24)throw Error("Missing immutable source plan");
const out=path.join(root,"screenshots");
if(fs.readdirSync(out).length>0 ||fs.existsSync(path.join(root,"capture.json")))throw Error("Refuse overwrite/relaunch");
const hash=b=>crypto.createHash("sha256").update(b).digest("hex");
const edge="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
if(!fs.existsSync(edge))throw Error("Edge executable missing");
async function main(){
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
    "--no-first-run","--disable-extensions","--disable-background-networking",
    "--disable-sync","--disable-component-update"]});
 try{
  const browserVersion=await browser.version();
  const page=await browser.newPage();
  await page.setViewport({width:960,height:540,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on("request",r=>r.url().startsWith("file:")?r.continue():r.abort());
  const receipts=[];
  let total=0;
  async function snapshot(scene,label,action){
    const state=await page.evaluate(()=>{
      const d=document.body.dataset;
      const controls=[...document.querySelectorAll("[data-ref]")].map(el=>{
       const r=el.getBoundingClientRect();
       return {ref:el.dataset.ref,role:el.tagName==="INPUT"?"Edit":"Button",
        name:el.getAttribute("aria-label"),bounds:[r.x,r.y,r.width,r.height],
        visible:r.width>0&&r.height>0};
      });
      return {selected:d.h45Selected==="null"?null:d.h45Selected,
        amount:d.h45Amount==="null"?null:Number(d.h45Amount),
        confirmed:d.h45Confirmed==="true",unrelated_changed:d.h45UnrelatedChanged==="true",
        controls,
        display:document.getElementById("h45-status").textContent,
        lang:document.documentElement.lang,rtl:document.documentElement.dir==="rtl"};
    });
    const file=scene.id+"-"+label+".png";
    const dest=path.join(out,file);
    if(fs.existsSync(dest))throw Error("Never overwrite PNG "+file);
    const png=Buffer.from(await page.screenshot({type:"png",fullPage:false,
              animations:"disabled",caret:"hide"}));
    if(png.subarray(0,8).toString("hex")!=="89504e470d0a1a0a"
       ||png.readUInt32BE(16)!==960||png.readUInt32BE(20)!==540)throw Error("Bad actual browser screenshot");
    fs.writeFileSync(dest,png,{flag:"wx"});
    total++;
    return {label,action,filename:file,image_sha256:hash(png),image_bytes:png.length,...state};
  }
  async function doAction(action){
    const ref=action.ref;
    const locator='[data-ref="'+ref+'"]';
    if(!["row_target","row_decoy","confirm_button","amount_field","unrelated_button"].includes(ref))
      throw Error("Unexpected H45 source action");
    if(action.kind==="press"){
      if(action.value!==null ||ref==="amount_field")throw Error("Unexpected click action");
      await page.click(locator);
    }else if(action.kind==="fill" && ref==="amount_field" && Number.isInteger(action.value)){
      await page.click(locator);
      await page.type(locator,String(action.value),{delay:0});
    }else throw Error("Invalid simulated source action");
  }
  for(let i=0;i<plan.scenes.length;i++){
    const scene=plan.scenes[i],task=orig.cases[i];
    if(scene.id!==task.id || scene.language!==task.language)throw Error("task/scene mismatch");
    const file=path.join(root,"pages",scene.id+".html");
    const bytes=fs.readFileSync(file);
    if(hash(bytes)!==scene.original_html_sha256)throw Error("Original H45 HTML bytes changed");
    async function navigate(){
      await page.goto(pathToFileURL(file).href,{waitUntil:"load",timeout:15000});
    }
    await navigate();
    const states=[await snapshot(scene,"step0",null)];
    for(let j=0;j<task.golden_actions.length;j++){
      const action=task.golden_actions[j];
      await doAction(action);
      states.push(await snapshot(scene,"step"+(j+1),action));
    }
    // Negative 1: wrong item, same edit + confirm.
    await navigate();
    const badActions=[{kind:"press",ref:"row_decoy",value:null},
                       task.golden_actions[1],task.golden_actions[2]];
    for(const action of badActions)await doAction(action);
    const wrong=await snapshot(scene,"negative_wrong_target",badActions);
    // Negative 2: complete correct goal but mutate unrelated entity.
    await navigate();
    const unrelated=[...task.golden_actions,{kind:"press",ref:"unrelated_button",value:null}];
    for(const action of unrelated)await doAction(action);
    const mutated=await snapshot(scene,"negative_unrelated_mutation",unrelated);
    receipts.push({id:scene.id,family:scene.family,language:scene.language,
       split:scene.split,page_sha256:hash(bytes),states,negatives:[wrong,mutated]});
  }
  const data={schema:"hala.h45.actual-edge-interactions.v1",
     browser_version:browserVersion,puppeteer_core:ver,
     viewport:{width:960,height:540,device_scale_factor:1},
     h45_scene_sha256:hash(fs.readFileSync(path.join(root,"scenes.json"))),
     original_h43_manifest_sha256:hash(fs.readFileSync(path.resolve("h43/data/pilot_manifest.json"))),
     screenshot_count:total,actual_browser_ui_interactions:true,
     only_first_party_synthetic_pages:true,
     external_network_requests:false,actual_model_inference:false,
     external_benchmark:false,native_desktop_actions:false,
     records:receipts};
  fs.writeFileSync(path.join(root,"capture.json"),JSON.stringify(data,null,2)+"\n",{flag:"wx"});
  console.log("H45_ACTUAL_EDGE_INTERACTIONS",receipts.length,"CASE_COUNT",total,
              "VERSION",browserVersion,"SOURCE_HASH",hash(fs.readFileSync(path.join(root,"capture.json"))));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1});
