"use strict";
// H137: independently implemented open-source UI engines, owner-authored harness.
// All three interaction behaviours come from pinned third-party MIT libraries.
const params=new URLSearchParams(location.search);
const caseId=Number(params.get("id"));
if(!Number.isSafeInteger(caseId)||caseId<570000||caseId>579999)throw Error("H137 invalid case id");
const family=caseId%3;
const arabic=caseId%2===1;
const workspace=document.querySelector("#workspace");
const root=document.querySelector("#ui-root");
const info=document.querySelector("#instruction");
const status=document.querySelector("#status");
const styles=["theme-dark","theme-orange","theme-green"];
document.body.classList.add(styles[Math.floor(caseId/3)%3]);
const cardNames=arabic?["مراجعة","تسليم","مهمة جديدة","قائمة"]:["Review","Delivery","Backlog","Operations"];
function el(name,attrs={},children=""){
  const elem=document.createElement(name);
  for(const [key,value] of Object.entries(attrs)){
    if(key==="className")elem.className=value;
    else if(key==="style")Object.assign(elem.style,value);
    else elem.setAttribute(key,String(value));
  }
  if(children)elem.textContent=children;
  return elem;
}
function marker(x,y,w=22,h=48){
  const e=el("div",{id:"end-target",style:{left:(x-w/2)+"px",top:(y-h/2)+"px",width:w+"px",height:h+"px"}});
  workspace.appendChild(e);
  return e;
}
function complete(yes){
  if(yes){workspace.dataset.success="true";status.textContent="Completed";}
}
const descriptions=[
  ["اسحب مقبض شريط التقدّم إلى العلامة الذهبية المتقطعة.",
   "Drag the progress slider thumb to the dashed gold checkpoint."],
  ["اسحب الفاصل بين اللوحتين إلى خط الهدف المتقطع.",
   "Drag the divider between the two panels to the marked checkpoint."],
  ["اسحب البطاقة الأولى من القائمة اليسرى إلى منطقة التسليم اليمنى.",
   "Drag the first task card from the left list into the highlighted delivery list."]
];
info.textContent=descriptions[family][arabic?0:1];
document.documentElement.lang=arabic?"ar":"en";
const hint=el("div",{className:"hint",style:{left:"70px",top:"66px"}},family===0?"Progress":family===1?"Panels":"Workflow");
workspace.appendChild(hint);
if(family===0){
  if(typeof noUiSlider==="undefined")throw Error("H137 genuine noUiSlider runtime absent");
  const rail=el("div",{id:"slider-track"});workspace.appendChild(rail);
  const start=18+(Math.floor(caseId/6)%5)*7;
  const target=72+(Math.floor(caseId/12)%3)*5;
  noUiSlider.create(rail,{start:[start],connect:[true,false],
    range:{min:0,max:100},step:1,behaviour:"tap-drag"});
  const handle=rail.querySelector(".noUi-handle");
  if(!handle)throw Error("Real noUiSlider handle was not created");
  handle.id="start-target";
  const b=rail.getBoundingClientRect(),outer=workspace.getBoundingClientRect();
  marker(b.left-outer.left+b.width*target/100, b.top-outer.top+b.height/2,22,64);
  workspace.dataset.vendor="nouislider";
  rail.noUiSlider.on("change",()=>{
    const current=Number(rail.noUiSlider.get());
    complete(Math.abs(current-target)<=4);
  });
}else if(family===1){
  if(typeof Split==="undefined")throw Error("H137 genuine Split.js runtime absent");
  const shell=el("div",{id:"split-shell",style:{display:"flex"}});
  const a=el("div",{id:"split-a",className:"split-pane"},"Overview");
  const b=el("div",{id:"split-b",className:"split-pane"},"Detail view");
  shell.append(a,b);workspace.appendChild(shell);
  const initial=30+(Math.floor(caseId/6)%3)*5;
  const target=62+(Math.floor(caseId/9)%4)*4;
  const widget=Split(["#split-a","#split-b"],{
    sizes:[initial,100-initial],minSize:65,gutterSize:14,
    onDragEnd:()=>complete(Math.abs(widget.getSizes()[0]-target)<=4)
  });
  const handle=shell.querySelector(".gutter");
  if(!handle)throw Error("Real Split.js gutter not created");
  handle.id="start-target";
  const box=shell.getBoundingClientRect(),wrap=workspace.getBoundingClientRect();
  marker(box.left-wrap.left+box.width*target/100,box.top-wrap.top+box.height/2,20,230);
  workspace.dataset.vendor="splitjs";
}else{
  if(typeof Sortable==="undefined")throw Error("H137 genuine SortableJS runtime absent");
  const shell=el("div",{id:"sortable-shell",style:{display:"flex"}});
  const left=el("div",{id:"source-list",className:"sort-column"});
  const right=el("div",{id:"destination-list",className:"sort-column"});
  for(let i=0;i<3;i++){
    const card=el("div",{className:"card","data-id":"source-"+i});
    const grip=el("span",{className:"grip"}, "⋮⋮");
    if(i===0)grip.id="start-target";
    card.append(grip,el("span",{},cardNames[(i+caseId)%4]));
    left.appendChild(card);
  }
  right.appendChild(el("div",{className:"card","data-id":"existing"},"✓ "+cardNames[(caseId+2)%4]));
  const mark=el("div",{id:"end-target",style:{position:"relative",left:"auto",top:"auto",height:"110px",width:"190px",marginTop:"10px"}});
  right.appendChild(mark);
  shell.append(left,right);workspace.appendChild(shell);
  const leftOptions={group:"h137owner",animation:0,handle:".grip",forceFallback:true,
                     draggable:".card",fallbackTolerance:0,fallbackOnBody:true};
  new Sortable(left,leftOptions);
  new Sortable(right,{group:"h137owner",animation:0,forceFallback:true,draggable:".card",
    onAdd:(evt)=>complete(evt.item.getAttribute("data-id")==="source-0")});
  workspace.dataset.vendor="sortablejs";
}
if(!document.querySelector("#start-target")||!document.querySelector("#end-target"))
  throw Error("H137 independent live start/destination controls absent");
