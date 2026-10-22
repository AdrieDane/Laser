"use strict";
const $ = id => document.getElementById(id);
const token = location.pathname.split("/")[1];
let state = null, lastPlan = null, selected = new Set(), dirty = false;
let refreshing = false, connected = false, receivedAt = 0;
function duration(seconds){const s=Math.max(0,Math.floor(seconds));return s<60?s+"s":s<3600?Math.floor(s/60)+"m "+s%60+"s":Math.floor(s/3600)+"h "+Math.floor(s%3600/60)+"m";}
function activity(){
 if(!state)return;
 const a=state.activity||{},running=busy(),extra=running&&receivedAt?(performance.now()-receivedAt)/1000:0;
 const age=(state.progress_age_seconds||0)+extra;
 $("activity").hidden=state.status==="idle";
 $("activity-phase").textContent=a.phase||state.status;
 $("activity-path").textContent=a.current_path||"Waiting for the next operation…";
 $("activity-elapsed").textContent=duration((state.elapsed_seconds||0)+extra);
 $("activity-age").textContent=duration(age)+" ago";
 $("activity-health").textContent=!connected?"Local scanner not responding":running?state.worker_alive?"Scanner responding · background task running":"Scanner responding · background task stopped":"Scanner responding · task "+state.status;
 $("activity-health").className=connected?"hint":"hint unreachable";
 const counts=[];
 for(const [key,label]of [["folders_checked","folders checked"],["files_seen","files seen"],["projects_found","projects found"],["relevant_files_found","relevant files found"]])if(a[key]!==undefined)counts.push(Number(a[key]).toLocaleString()+" "+label);
 if(a.datasets_total!==undefined)counts.push(a.datasets_done+" / "+a.datasets_total+(["copying","done"].includes(state.status)?" datasets copied":" datasets checked"));
 if(a.files_total!==undefined)counts.push(a.files_done+" / "+a.files_total+" files checked");
 if(a.annotations_total!==undefined&&a.phase==="Matching curated annotations")counts.push(a.annotations_checked+" / "+a.annotations_total+" annotations checked");
 $("activity-counts").textContent=counts.join(" · ");
 const bar=$("activity-progress");bar.hidden=true;
 let label=a.phase==="Finding project folders"?"The total number of folders is not known yet; no overall percentage is estimated.":"";
 if(a.file_bytes_total>0){bar.hidden=false;bar.max=a.file_bytes_total;bar.value=a.file_bytes_done||0;label="Current file: "+size(a.file_bytes_done||0)+" / "+size(a.file_bytes_total);}
 else if(a.datasets_total>0){bar.hidden=false;bar.max=a.datasets_total;bar.value=a.datasets_done||0;label="Dataset checks: "+a.datasets_done+" / "+a.datasets_total;}
 $("activity-progress-label").textContent=label;
 let warning="";
 if(!connected)warning="The browser cannot reach the local scanner. Keep the scanner console window open. This is a connection problem, not proof that the source scan has stopped.";
 else if(running&&state.cancel_requested)warning="Cancellation requested. Waiting for the current operation to return. Windows network reads cannot always be interrupted immediately; source files remain unchanged.";
 else if(running&&age>=30)warning=(age>=120?"Possible stall: ":"")+"no progress update for "+duration(age)+". The last operation is shown above. It may be waiting for the network drive or reading a large file; a responding scanner does not prove the file operation is progressing.";
 $("activity-warning").hidden=!warning;$("activity-warning").textContent=warning;
 $("activity-log").textContent=(state.recent_activity||[]).map(e=>duration(e.elapsed_seconds)+"  "+e.message+(e.path?"\n    "+e.path:"")).join("\n");
}
const fields = {
  min_annotations:["Minimum curated entries per mode",1], min_classes:["Minimum lipid classes per mode",1],
  min_coverage:["Minimum measurable / matched fraction (0–1)",.01], max_ppm:["Absolute mass-error limit (ppm)",.1],
  max_rt_seconds:["Absolute RT-error limit (seconds)",1], min_within_limits:["Minimum fraction within error limits (0–1)",.01],
  max_median_ppm:["Maximum median absolute mass error (ppm)",.1], max_median_rt_seconds:["Maximum median absolute RT error (seconds)",.5],
  min_profile_correlation:["Minimum sample-profile correlation (0–1)",.001], min_profile_samples:["Minimum matched profile samples",1]
};
function text(tag,value,className){const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;}
function size(n){for(const unit of ["B","KiB","MiB","GiB","TiB"]){if(n<1024||unit==="TiB")return n.toLocaleString(undefined,{maximumFractionDigits:1})+" "+unit;n/=1024;}}
function fmt(n,unit=""){return n===undefined||n===null?"Not assessed":n.toFixed(2)+unit;}
async function post(path,body={}){const r=await fetch("api/"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Scanner-Token":token},body:JSON.stringify(body)});const value=await r.json();if(!r.ok)throw Error(value.error||"Request failed");return value;}
function fail(e){$("message").textContent=e.message;}
function busy(){return state&&["scanning","copying"].includes(state.status);}
function totals(){const plan=state&&state.plan;const ds=plan?plan.datasets.filter(d=>selected.has(d.id)):[];$("file-count").textContent=ds.reduce((n,d)=>n+d.files.length,0);$("copy-size").textContent=size(ds.reduce((n,d)=>n+d.total_bytes,0));$("copy").disabled=busy()||dirty||!ds.length||!$("approval").checked||!$("destination").value.trim();}
function table(headers,rows){const t=document.createElement("table"),head=document.createElement("thead"),tr=document.createElement("tr");headers.forEach(h=>tr.append(text("th",h)));head.append(tr);t.append(head);const b=document.createElement("tbody");rows.forEach(row=>{const r=document.createElement("tr");row.forEach(v=>r.append(text("td",v)));b.append(r);});t.append(b);return t;}
function render(plan){$("results").hidden=false;$("copy-stage").hidden=false;$("approval").checked=false;$("issues").textContent=plan.discovery_issues.join("; ");$("pass-count").textContent=plan.datasets.filter(d=>d.status==="PASS").length;$("datasets").replaceChildren();
 for(const d of plan.datasets){const item=document.createElement("details");item.className="dataset";const summary=document.createElement("summary"),label=document.createElement("label"),check=document.createElement("input");check.type="checkbox";check.checked=selected.has(d.id);check.disabled=d.status!=="PASS";check.addEventListener("click",e=>e.stopPropagation());check.addEventListener("change",()=>{if(check.checked)selected.add(d.id);else selected.delete(d.id);$("approval").checked=false;totals();});label.append(check,document.createTextNode(" "+d.project));summary.append(label,text("span",d.status+" · "+size(d.total_bytes),"status "+d.status.toLowerCase()));item.append(summary,text("p",d.source_relative,"hint"));
 item.append(table(["Mode","Status","Curated entries","Matched original features","Mass < limit","Median |ppm|","RT within limit","Median |RT|"],d.modes.map(m=>[m.stream,m.status,m.curated_entries??"—",fmt(m.feature_match_percent,"%"),fmt(m.mass_within_limit_percent,"%"),fmt(m.median_absolute_ppm),fmt(m.rt_within_limit_percent,"%"),fmt(m.median_absolute_rt_seconds," s")])));
 if(d.reasons.length){const list=document.createElement("ul");list.className="reasons";d.reasons.forEach(r=>list.append(text("li",r)));item.append(list);}
 item.append(text("h3","Exact files selected for collection"));item.append(table(["Original relative path","Reason","Size"],d.files.map(f=>[f.source_relative,f.role,size(f.size)])));$("datasets").append(item);
 }if(!plan.datasets.length)$("datasets").append(text("p","No eligible project folders or curated annotation workbooks were found."));totals();}
async function refresh(){if(refreshing)return;refreshing=true;const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),5000);try{const r=await fetch("api/status",{signal:controller.signal});if(!r.ok)throw Error("Cannot reach the local scanner");state=await r.json();connected=true;receivedAt=performance.now();if(!$("source").value)$("source").value=state.default_source;if(!$("limits").children.length)for(const [key,[label,step]]of Object.entries(fields)){const l=text("label",label),i=document.createElement("input");i.type="number";i.step=step;i.min=step;i.value=state.default_limits[key];i.id="limit-"+key;if(key.includes("fraction")||["min_coverage","min_within_limits","min_profile_correlation"].includes(key))i.max=1;i.addEventListener("input",markDirty);l.append(i);$("limits").append(l);}
 $("message").textContent=dirty&&!busy()?"Source or limits changed. Run stage 1 again before copying.":state.message;$("copy-message").textContent=["copying","done","error","cancelled"].includes(state.status)?state.message:"";$("reports").textContent=state.reports?"Detailed report saved on this PC: "+state.reports:"";$("scan").disabled=busy();$("cancel").disabled=!busy();$("source").disabled=busy();document.querySelectorAll("#limits input").forEach(i=>i.disabled=busy());$("completed").textContent=(state.copied||[]).join("\n");if(state.plan&&lastPlan!==state.plan.scan_id){lastPlan=state.plan.scan_id;dirty=false;selected=new Set(state.plan.datasets.filter(d=>d.status==="PASS").map(d=>d.id));render(state.plan);}totals();
 }catch(e){connected=false;fail(Error(e.name==="AbortError"?"Local scanner has not responded for 5 seconds. Check its console window.":e.message));}finally{clearTimeout(timeout);refreshing=false;activity();}}
function markDirty(){if(state&&state.plan){dirty=true;$("approval").checked=false;$("message").textContent="Source or limits changed. Run stage 1 again before copying.";totals();}}
$("source").addEventListener("input",markDirty);$("destination").addEventListener("input",totals);$("approval").addEventListener("change",totals);
document.querySelectorAll("[data-browse]").forEach(b=>b.addEventListener("click",async()=>{try{const r=await post("pick-folder");if(r.path){$(b.dataset.browse).value=r.path;if(b.dataset.browse==="source")markDirty();totals();}}catch(e){fail(e);}}));
$("scan").addEventListener("click",async()=>{try{const limits=Object.fromEntries(Object.keys(fields).map(k=>[k,Number($("limit-"+k).value)]));await post("scan",{source:$("source").value.trim(),limits});dirty=true;$("approval").checked=false;await refresh();}catch(e){fail(e);}});
$("copy").addEventListener("click",async()=>{try{if(!$("approval").checked||dirty)return;await post("copy",{destination:$("destination").value.trim(),dataset_ids:[...selected],confirm:true,scan_id:state.plan.scan_id});$("approval").checked=false;await refresh();}catch(e){fail(e);}});
$("cancel").addEventListener("click",async()=>{try{await post("cancel");await refresh();}catch(e){fail(e);}});$("stop").addEventListener("click",async()=>{if(busy()&&!confirm("Cancel the current task and close the scanner? Incomplete copies will remain in .partial_ folders."))return;await post("stop");$("message").textContent="Scanner closed. You can close this browser tab.";clearInterval(timer);clearInterval(activityTimer);});
refresh();const timer=setInterval(refresh,1200),activityTimer=setInterval(activity,1000);
