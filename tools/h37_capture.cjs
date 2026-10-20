// Read-only offline Chromium fixture capture; no native actions or external sites.
// Requires caller-installed exact playwright-core version. Output is create-only.
"use strict";
const fs=require("node:fs");
const path=require("node:path");
const crypto=require("node:crypto");
const {pathToFileURL}=require("node:url");

function sha(bytes){return crypto.createHash("sha256").update(bytes).digest("hex");}
async function main(){
  const [source, modulePath, browserPath] = process.argv.slice(2);
  if(!source || !modulePath || !browserPath) throw Error("usage: node h37_capture.cjs <new H37 scenes dir> <playwright-core module path> <browser exe>");
  const pw=require(modulePath);
  const pkg=require(path.join(modulePath, "package.json"));
  if(pkg.version!=="1.56.1")throw Error("playwright-core pin mismatch");
  const root=path.resolve(source);
  const scenes=JSON.parse(fs.readFileSync(path.join(root,"scenes.json"),"utf8"));
  const meta=JSON.parse(fs.readFileSync(path.join(root,"meta.json"),"utf8"));
  const screenshotDir=path.join(root,"screenshots");
  if(fs.existsSync(screenshotDir) || fs.existsSync(path.join(root,"capture.json")))throw Error("H37 outputs already exist");
  fs.mkdirSync(screenshotDir);
  const browser=await pw.chromium.launch({
     executablePath:browserPath, headless:true,
     args:["--disable-background-networking","--disable-extensions","--no-first-run","--no-default-browser-check"],
  });
  const browserVersion=browser.version();
  const records=[];
  try{
    for(const scene of scenes){
      const context=await browser.newContext({
        viewport:{width:960,height:540},deviceScaleFactor:1,
        colorScheme:"light",reducedMotion:"reduce",locale:scene.language==="ar"?"ar-SA":"en-US"
      });
      await context.setOffline(true);
      const page=await context.newPage();
      const file=path.join(root,"pages",scene.id+".html");
      const bytes=fs.readFileSync(file);
      if(sha(bytes)!==meta.html_sha256[scene.id])throw Error("H37 HTML hash mismatch");
      await page.goto(pathToFileURL(file).href,{waitUntil:"load",timeout:15000});
      if(await page.locator("script").count())throw Error("Script content not permitted");
      if((await page.locator("button[data-control-id]").count())!==4)throw Error("Wrong visible controls");
      const controls=await page.locator("button[data-control-id]").evaluateAll(nodes=>nodes.map(node=>{
        const bounds=node.getBoundingClientRect();
        return {
          id:node.getAttribute("data-control-id"),
          label:node.textContent.trim(),
          bounds:[bounds.x,bounds.y,bounds.width,bounds.height].map(x=>Math.round(x*100)/100),
          visible:bounds.width>0&&bounds.height>0&&getComputedStyle(node).visibility!=="hidden"
        };
      }));
      if(controls.some(x=>!x.visible)||new Set(controls.map(x=>x.id)).size!==4){
        throw Error("Hidden/duplicate H37 browser targets");
      }
      const image=path.join(screenshotDir,scene.id+".png");
      const png=await page.screenshot({path:image,type:"png",animations:"disabled",caret:"hide"});
      if(png.subarray(0,8).toString("hex")!=="89504e470d0a1a0a")throw Error("Invalid PNG screenshot");
      records.push({
        id:scene.id,split:scene.split,language:scene.language,layout:scene.layout,
        family:scene.family,variant:scene.variant,width:960,height:540,
        goal:scene.goal,expected_control_id:scene.expected_control_id,
        controls,image:scene.id+".png",image_sha256:sha(png),
        bytes:png.length,html_sha256:sha(bytes)
      });
      await context.close();
    }
  }finally{
    await browser.close();
  }
  if(records.length!==64)throw Error("Incomplete browser capture");
  const data={
    schema:"hala.h37.browser_capture.v1",
    renderer:"real locally installed Chromium-based Microsoft Edge, Playwright-core",
    browser_version:browserVersion,playwright_core:pkg.version,
    device_pixel_ratio:1,width:960,height:540,rendered_records:records,
    cases_sha256:meta.scenes_sha256,
    native_actions_executed:false,model_inference:false,origin:"file:// local first-party authored HTML, offline"
  };
  const output=path.join(root,"capture.json");
  fs.writeFileSync(output,JSON.stringify(data,null,2)+"\n",{flag:"wx"});
  console.log(JSON.stringify({records:records.length,browserVersion,
    playwright:pkg.version,capture_sha256:sha(fs.readFileSync(output)),
    uniqueImages:new Set(records.map(x=>x.image_sha256)).size}));
}
main().catch(err=>{console.error("H37_BROWSER_CAPTURE_FAILED:",err.message);process.exitCode=1;});
