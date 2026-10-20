/* H48 actual chromium/Edge original train-only GUI screenshot recorder.
 * Never touches real sites, user apps, accounts, model benchmarks or GPU.
 * Training-only source and authoritative DOM labels are retained in Git.
 */
"use strict";
const fs=require("fs"),path=require("path"),crypto=require("crypto");
const {pathToFileURL}=require("url");
const modulePath=process.env.PUPPETEER_CORE_PATH;
if(!modulePath || !fs.existsSync(path.join(modulePath,"package.json")))
  throw Error("Pinned existing local puppeteer-core required");
if(JSON.parse(fs.readFileSync(path.join(modulePath,"package.json"),"utf8")).version!=="25.12.0")
  throw Error("Puppeteer version changed");
const puppeteer=require(modulePath);
const root=path.resolve(process.argv[2]||"h48/assets-v2");
const sourceFile=path.join(root,"source_manifest.json");
const original=JSON.parse(fs.readFileSync(sourceFile,"utf8"));
if(original.schema!=="hala.h48.original-train-only-visual-scenes.v2" ||
   original.sources.length!==128 || original.train_only!==true)throw Error("Wrong original H48 source set");
const shots=path.join(root,"screenshots");
if(fs.existsSync(shots)&&fs.readdirSync(shots).length)throw Error("Refuse overwriting H48 pixel originals");
fs.mkdirSync(shots,{recursive:true});
const sha=b=>crypto.createHash("sha256").update(b).digest("hex");
const edge="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
if(!fs.existsSync(edge))throw Error("Actual Edge binary unavailable");
(async()=>{
 const browser=await puppeteer.launch({executablePath:edge,headless:true,args:[
   "--no-first-run","--disable-extensions","--disable-background-networking",
   "--disable-sync","--disable-component-update"]});
 const receipts=[];
 try{
  const browserVersion=await browser.version();
  const page=await browser.newPage();
  await page.setViewport({width:960,height:540,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on("request",req=>req.url().startsWith("file:")?req.continue():req.abort());
  for(let i=0;i<original.sources.length;i++){
    const row=original.sources[i];
    const html=fs.readFileSync(path.join(root,"pages",row.id+".html"));
    if(sha(html)!==row.source_sha256)throw Error("Unpinned source "+row.id);
    await page.goto(pathToFileURL(path.join(root,"pages",row.id+".html")).href,
                    {waitUntil:"load",timeout:15000});
    const snapshot=await page.evaluate(()=>({
      language:document.documentElement.lang,
      direction:document.documentElement.dir,
      elements:[...document.querySelectorAll("button[data-option]")].map(b=>{
       const r=b.getBoundingClientRect();
       return {option_index:Number(b.dataset.option),name:b.getAttribute("aria-label"),
         rect:[r.x,r.y,r.width,r.height],visible:r.width>0&&r.height>0};})
    }));
    if(snapshot.elements.length!==4||snapshot.language!==row.language||
       snapshot.direction!==(row.language==="ar"?"rtl":"ltr"))
       throw Error("Wrong bilingual original browser controls "+row.id);
    const dest=path.join(shots,row.id+".png");
    if(fs.existsSync(dest))throw Error("No overwrite original "+dest);
    const png=Buffer.from(await page.screenshot({type:"png",fullPage:false,
                    animations:"disabled",caret:"hide"}));
    if(png.subarray(0,8).toString("hex")!=="89504e470d0a1a0a"||
       png.readUInt32BE(16)!==960||png.readUInt32BE(20)!==540)
       throw Error("Invalid real browser image");
    fs.writeFileSync(dest,png,{flag:"wx"});
    receipts.push({id:row.id,image_sha256:sha(png),image_bytes:png.length,
       source_sha256:row.source_sha256,language:snapshot.language,controls:snapshot.elements});
    if(i%32===31)console.log("H48_SOURCE_CAPTURE_PROGRESS",i+1,"of",128);
  }
  const audit={schema:"hala.h48.actual-chromium-128-train-images.v1",
    chromium_version:browserVersion,puppeteer_core:"25.12.0",
    source_manifest_sha256:sha(fs.readFileSync(sourceFile)),
    width:960,height:540,train_only:true,benchmark:false,model_inference:false,
    external_network_requests:false,native_actions:false,records:receipts};
  fs.writeFileSync(path.join(root,"capture.json"),JSON.stringify(audit,null,2)+"\n",{flag:"wx"});
  console.log("H48_ACTUAL_ORIGINAL_BROWSER_IMAGES",receipts.length,
              "HASH",sha(fs.readFileSync(path.join(root,"capture.json"))));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1});
