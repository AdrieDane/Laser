const node = (tag, text='', className='') => {
  const element=document.createElement(tag); if(text) element.textContent=text;
  if(className) element.className=className; return element;
};
const numberText = (value, digits=3) => value==null || !Number.isFinite(Number(value))?'—':Number(value).toFixed(digits);
const plotFont = {family:'Arial, sans-serif',size:12};
const traceOrder=['background','significant','adduct','cluster','significantCluster','focus'];
const legendNames={background:'Not significant',significant:'Other significant features',adduct:'Other adduct candidate',
  cluster:'Cluster member',significantCluster:'Significant cluster member',focus:'Selected / possible counterpart'};

export function modeFigure(state) {
  const {model}=state, context=model.context(state.selected), visible=model.visibleStreams(state.view);
  const phases=[...new Set(visible.map(stream=>stream.phase))];
  const traces=[], shapes=[], annotations=[];
  const visibleKeys=new Set();
  for(const key of model.hits) if(visible.some(stream=>key.startsWith(stream.name+'\u0000'))) visibleKeys.add(key);
  for(const group of [context.focus,context.cluster,context.adducts]) for(const key of group) visibleKeys.add(key);
  if(state.grey) for(const stream of visible) for(const index of stream.background) visibleKeys.add(featureKey(stream.name,stream.ids[index]));
  const layout={height:phases.length>1?610:490, margin:{l:65,r:12,t:55,b:48},
    paper_bgcolor:state.canvas,plot_bgcolor:state.canvas,font:{...plotFont,color:state.ink},
    showlegend:true,legend:{orientation:'h',x:0,y:1.10,font:{size:10}},hovermode:'closest',
    uirevision:state.view,dragmode:'pan',shapes,annotations};
  phases.forEach((phase,panel)=>{
    const suffix=panel?String(panel+1):'', xaxis='x'+suffix,yaxis='y'+suffix;
    const buckets=new Map(traceOrder.map(role=>[role,[]]));
    for(const key of visibleKeys) {
      const point=model.point(key); if(!point || point.phase!==phase) continue;
      if(!visible.some(stream=>stream.name===point.stream)) continue;
      buckets.get(model.style(key,context)).push(point);
    }
    traceOrder.forEach(role=>{
      const points=buckets.get(role), colour=MODE_COLOURS[role];
      traces.push({type:'scattergl',mode:'markers',xaxis,yaxis,name:legendNames[role],legendgroup:role,
        showlegend:panel===0, x:points.map(p=>p.rt), y:points.map(p=>p.negative?-p.mz:p.mz),
        customdata:points.map(p=>[p.key,p.stream,p.id,p.mz,model.rows.get(p.key)?.compound_id??'',
          model.rows.get(p.key)?.adduct??'',model.rows.get(p.key)?.p_value??null]),
        marker:{color:colour,opacity:role==='background'?.32:1,size:role==='background'?4:role==='significant'?7:11,
          symbol:role==='significant'?'diamond':'circle',line:{color:role==='significantCluster'?MODE_COLOURS.cluster:colour,
            width:role==='significantCluster'?3:role==='focus'?2:0}},
        hovertemplate:'%{customdata[4]}<br>%{customdata[1]} · feature %{customdata[2]}<br>'+
          'm/z %{customdata[3]:.5f} · RT %{x:.3f} min<br>%{customdata[5]}<br>raw p %{customdata[6]:.3g}<extra>'+legendNames[role]+'</extra>'});
    });
    let maxRT=1,maxMZ=100;
    for(const stream of visible.filter(item=>item.phase===phase)) {
      maxMZ=Math.max(maxMZ,stream.mz[stream.mz.length-1]??100);
      for(const value of stream.rt) if(value>maxRT) maxRT=value;
    }
    const common={gridcolor:state.grid,zeroline:false,linecolor:state.grid,fixedrange:false};
    layout['xaxis'+suffix]={...common,anchor:yaxis,domain:[0,1],title:{text:panel===phases.length-1?'Retention time (min)':''},range:[0,maxRT*1.02]};
    layout['yaxis'+suffix]={...common,anchor:xaxis,title:{text:'m/z (+POS / −NEG)'},
      domain:phases.length===1?[0,1]:panel===0?[.57,1]:[0,.43],range:[-maxMZ*1.04,maxMZ*1.04]};
    shapes.push({type:'line',xref:xaxis+' domain',yref:yaxis,x0:0,x1:1,y0:0,y1:0,line:{color:state.ink,width:1}});
    if(phases.length>1) annotations.push({xref:'paper',yref:'paper',x:0,y:panel===0?1.01:.45,text:phase,showarrow:false,font:{size:13,color:state.ink}});
  });
  return {data:traces,layout};
}

export function sampleFigure(state) {
  const {model}=state, bars=model.bars(state.selected), row=model.rows.get(state.selected);
  const traces=[];
  if(bars) {
    const groups=[...new Set(bars.stream.groups)].sort();
    for(const group of groups) {
      const indices=bars.samples.map((_,index)=>index).filter(index=>bars.stream.groups[index]===group && Number.isFinite(bars.values[index]));
      const label=bars.stream.groupLabels[group]??group, lowered=(label+' '+group).toLowerCase();
      const colour=/blank|blanco/.test(lowered)?'#c7c7cc':/qc|pool/.test(lowered)||group==='Unassigned'?'#8e8e93':model.data.palette[group];
      traces.push({type:'bar',name:label,x:indices,y:indices.map(index=>bars.values[index]),
        marker:{color:colour},customdata:indices.map(index=>bars.samples[index]),
        hovertemplate:'<b>%{customdata}</b><br>%{y:.3g}<extra>%{fullData.name}</extra>'});
    }
  }
  const title=row?row.compound_id+' · '+(row.adduct||'adduct unresolved'):'Select a significant feature';
  return {data:traces,layout:{height:280,margin:{l:54,r:20,t:38,b:45},title:{text:title,x:.5,xanchor:'center',font:{size:13}},
    paper_bgcolor:state.canvas,plot_bgcolor:state.canvas,font:{...plotFont,color:state.ink},
    xaxis:{showticklabels:false,ticks:'',gridcolor:state.grid},yaxis:{title:{text:'intensity'},rangemode:'tozero',tickformat:'.2s',gridcolor:state.grid},
    barmode:'overlay',bargap:.12,showlegend:true,legend:{orientation:'h',y:-.08,x:0,font:{size:10}}}};
}

function smallTable(parent, headers, rows) {
  parent.replaceChildren();
  if(!rows.length) {parent.appendChild(node('p','No candidates within the saved search tolerances.'));return;}
  const table=node('table'),head=node('tr'),body=node('tbody');
  for(const title of headers) head.appendChild(node('th',title));
  const thead=node('thead');thead.appendChild(head);table.append(thead,body);
  for(const values of rows.slice(0,250)) {
    const row=node('tr');for(const value of values) row.appendChild(node('td',String(value??'—')));body.appendChild(row);
  }
  parent.appendChild(table);
  if(rows.length>250) parent.appendChild(node('p',`Showing the first 250 of ${rows.length} candidates.`));
}

function renderEvidence(state) {
  state.details.style.display=state.selected?'':'none';
  if(!state.selected) return;
  const {model}=state,row=model.rows.get(state.selected),point=model.point(state.selected),context=model.context(state.selected);
  state.detailTitle.textContent=(row?.compound_id??'Feature')+' · '+(point?.stream??'')+' · feature '+(point?.id??'');
  state.detailFormula.textContent=row?.chemical_composition_hypothesis??'No isotope-supported formula is saved for this detection.';
  const isotope=[];
  for(const n of [1,2]) {
    const prefix='isotope_m_plus_'+n;
    if(row?.[prefix+'_feature_id']) isotope.push(['M+'+n,row[prefix+'_feature_id'],numberText(row[prefix+'_mz'],5),
      numberText(row[prefix+'_intensity_percent'],1),numberText(row[prefix+'_rt_delta_seconds'],2)]);
  }
  smallTable(state.isotopes,['Isotope','Feature','m/z','Intensity %','Δ RT (s)'],isotope);
  const detections=[...model.rows.values()].filter(item=>item.compound_id===row?.compound_id).map(item=>[
    item.feature_id,item.compound_id,item.cluster_id||'—',numberText(item.mz,5),numberText(item.rt,3),item.adduct||'unresolved',item.stream,
    item.raw_p_significant?'yes':'no',item.association_status]);
  smallTable(state.detections,['Feature','Compound','Cluster','m/z','RT','Likely adduct','Mode','Raw-p hit','Assignment'],detections);
  const clusterId=row?.cluster_id||[...context.focus].map(key=>model.clusterIds.get(key)).find(Boolean)||'';
  const clusters=[...context.cluster].map(key=>{const p=model.point(key);return p?[model.clusterIds.get(key)||clusterId||'candidate',p.stream,p.id,numberText(p.mz,5),numberText(p.rt,3),
    model.hits.has(key)?'yes':'no']:null;}).filter(Boolean);
  smallTable(state.clusters,['Cluster','Mode','Cluster feature','m/z','RT','Raw-p hit'],clusters);
  const matches=context.matches.map(link=>{const target=model.point(link.target);return target?[
    link.sourceAdduct,target.stream,target.id,link.targetAdduct,numberText(link.ppm,2),link.compared?numberText(link.rt,2):'not compared']:null;}).filter(Boolean);
  smallTable(state.adducts,['Source adduct','Target mode','Feature','Target adduct','Δ mass (ppm)','Δ RT (s)'],matches);
}

function renderRows(state, ensureSelected=false) {
  state.filtered=state.model.tableRows(state.view,state.filter.value);
  const height=31, viewport=state.tableViewport;
  if(ensureSelected) {
    const index=state.filtered.findIndex(([key])=>key===state.selected);
    if(index>=0 && (index*height<viewport.scrollTop || (index+1)*height>viewport.scrollTop+viewport.clientHeight)) viewport.scrollTop=Math.max(0,index*height-height*2);
  }
  state.tableBody.style.height=(height*state.filtered.length)+'px';
  const first=Math.max(0,Math.floor(viewport.scrollTop/height)-3),last=Math.min(state.filtered.length,first+Math.ceil((viewport.clientHeight||310)/height)+7);
  const fragment=document.createDocumentFragment();
  for(let index=first;index<last;index++) {
    const [key,row]=state.filtered[index],item=node('div','','feature-row');
    item.setAttribute('role','row'); item.tabIndex=0;item.style.top=(index*height)+'px';
    item.setAttribute('aria-selected',String(key===state.selected));item.dataset.feature=key.replace('\u0000','::');
    const values=[row.feature_id,row.compound_id,row.cluster_id||'—',numberText(row.mz,4),numberText(row.rt,3),row.adduct||'unresolved',row.stream,
      row.p_value==null?'—':Number(row.p_value).toExponential(2),row.raw_p_significant?'yes':'linked'];
    for(const value of values) {const cell=node('div',String(value),'feature-cell');cell.setAttribute('role','gridcell');cell.title=String(value);item.appendChild(cell);}
    item.addEventListener('click',()=>state.select(key,false));
    item.addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' ') {event.preventDefault();state.select(key,false);}
      if(event.key==='ArrowDown'||event.key==='ArrowUp') {event.preventDefault();const next=index+(event.key==='ArrowDown'?1:-1);if(state.filtered[next]) state.select(state.filtered[next][0],true);}
    });
    fragment.appendChild(item);
  }
  state.tableBody.replaceChildren(fragment);
  state.tableCount.textContent=state.filtered.length.toLocaleString()+' detections · likely adduct assignments are tentative';
}

function createWorkspace(parent, data) {
  const root=node('section','','mode-workspace');root.setAttribute('aria-label','Untargeted mode viewer');
  root.dataset.theme=data.theme;
  const controls=node('div','','mode-controls'),modes=node('div','','mode-buttons');modes.setAttribute('role','group');modes.setAttribute('aria-label','View modes');
  const buttons=new Map();
  for(const view of ['RP POS and NEG','NP POS and NEG','All 4 modes']) {const button=node('button',view);button.type='button';modes.appendChild(button);buttons.set(view,button);}
  const greyLabel=node('label','','grey-switch'),grey=node('input');grey.type='checkbox';grey.checked=false;
  greyLabel.title='Up to 20,000 background points. Related detections are always shown.';
  greyLabel.append(grey,document.createTextNode('Show non-significant grey points'));controls.append(modes,greyLabel);
  const top=node('div','','mode-top'),main=node('div','','mode-main'),plot=node('div','','mode-plot'),side=node('aside','','mode-side'),bar=node('div','','sample-bars');
  main.append(controls,plot);side.appendChild(bar);
  const legend=node('div','','colour-key');
  for(const [role,label] of [['focus','Selected / possible counterpart'],['cluster','Cluster member'],['adduct','Other adduct candidate'],['significantCluster','Significant cluster member']]) {
    const line=node('div'),swatch=node('span','','swatch');swatch.style.background=MODE_COLOURS[role];
    if(role==='significantCluster') swatch.style.border='3px solid '+MODE_COLOURS.cluster;
    line.append(swatch,document.createTextNode(label));legend.appendChild(line);
  }
  side.appendChild(legend);top.append(main,side);
  const tableTitle=node('h3','Significant compounds and linked detections'),tableTools=node('div','','table-tools'),filter=node('input');
  filter.type='search';filter.placeholder='Find compound, feature, m/z or mode';filter.setAttribute('aria-label','Find feature');
  const tableCount=node('span');tableTools.append(filter,tableCount);
  const table=node('div','','feature-grid');table.setAttribute('role','grid');table.setAttribute('aria-label','Significant features and linked detections');
  const header=node('div','','feature-header');header.setAttribute('role','row');
  for(const text of ['Feature','Compound','Cluster','m/z','RT (min)','Likely adduct','Mode','Raw p','Significant']) {const cell=node('div',text);cell.setAttribute('role','columnheader');header.appendChild(cell);}
  const tableViewport=node('div','','feature-viewport'),tableBody=node('div','','feature-body');tableViewport.appendChild(tableBody);table.append(header,tableViewport);
  const details=node('div','','mode-evidence'),detailTitle=node('h3');details.appendChild(detailTitle);
  const evidence={};
  for(const [key,title] of [['detections','Compound detections'],['isotopes','Isotopes and candidate formulas'],['clusters','Cluster members across modes'],['adducts','Adduct evidence']]) {
    const section=node('details'),summary=node('summary',title),content=node('div','','evidence-scroll');section.open=key==='isotopes';section.append(summary,content);details.appendChild(section);evidence[key]=content;
  }
  const detailFormula=node('p');evidence.isotopes.parentElement.appendChild(detailFormula);
  const review=node('button','Review selected feature','review-feature');review.type='button';details.appendChild(review);
  root.append(top,tableTitle,tableTools,table,details);parent.appendChild(root);
  const state={root,model:new ModeViewerModel(data),buttons,grey,plot,bar,filter,tableViewport,tableBody,tableCount,detailTitle,detailFormula,details,...evidence,
    version:0,busy:false,grey:false,view:Object.keys(data.streams).some(s=>s.startsWith('RP'))?'RP POS and NEG':'NP POS and NEG',timings:[]};
  state.canvas=data.theme==='Midnight'?'#20313f':'#fff';state.ink=data.theme==='Midnight'?'#e7edf2':'#1d1d1f';state.grid=data.theme==='Midnight'?'#3b5262':'#e1e6ef';
  state.selected=null;
  state.select=(key,scroll=true)=>{if(!state.model.rows.has(key)) return;state.selected=key;renderRows(state,scroll);state.requestedAt=performance.now();state.version++;updateWorkspace(state);};
  for(const [view,button] of buttons) button.addEventListener('click',()=>{state.view=view;renderRows(state,false);state.requestedAt=performance.now();state.version++;updateWorkspace(state);});
  grey.addEventListener('change',()=>{state.grey=grey.checked;state.requestedAt=performance.now();state.version++;updateWorkspace(state);});
  filter.addEventListener('input',()=>{tableViewport.scrollTop=0;renderRows(state,false);});
  tableViewport.addEventListener('scroll',()=>renderRows(state,false));
  review.addEventListener('click',()=>{if(state.selected) state.emit?.('review',state.selected.split('\u0000'));});
  state.observer=new ResizeObserver(()=>{if(state.plot._fullLayout) {window.__laserStandardPlotly.Plots.resize(state.plot);window.__laserStandardPlotly.Plots.resize(state.bar);}});
  state.observer.observe(top);
  return state;
}

async function updateWorkspace(state) {
  if(state.busy) return;
  state.busy=true;
  try {
    do {
      const version=state.version,start=state.requestedAt??performance.now();
      for(const [view,button] of state.buttons) button.setAttribute('aria-pressed',String(view===state.view));
      renderEvidence(state);
      const figure=modeFigure(state),bars=sampleFigure(state),Plotly=window.__laserStandardPlotly;
      await Promise.all([Plotly.react(state.plot,figure.data,figure.layout,{displaylogo:false,scrollZoom:true,responsive:true}),
        Plotly.react(state.bar,bars.data,bars.layout,{displayModeBar:false,responsive:true})]);
      if(!state.bound) {
        state.plot.on('plotly_click',event=>{const key=event.points?.[0]?.customdata?.[0];if(key&&state.model.rows.has(key)) state.select(key,true);});state.bound=true;
      }
      if(version===state.version) {
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        const elapsed=performance.now()-start;state.timings.push(Math.round(elapsed));state.timings=state.timings.slice(-60);
        state.root.dataset.updateMs=String(Math.round(elapsed));state.root.dataset.timings=JSON.stringify(state.timings);
        state.root.dataset.selection=state.selected?.replace('\u0000','::')??'';state.root.dataset.view=state.view;
        break;
      }
    } while(true);
  } catch(error) {state.root.appendChild(node('p','Viewer error: '+error.message,'viewer-error'));console.error(error);}
  finally {state.busy=false;}
}

export default function({data,parentElement,setTriggerValue}) {
  const host=parentElement.host??parentElement;
  let state=parentElement.__modeWorkspace;
  const ready=window.__laserStandardPlotly?Promise.resolve():new Promise(resolve=>document.addEventListener('laser-standard-plot-ready',resolve,{once:true}));
  ready.then(()=>{
    if(parentElement instanceof ShadowRoot && !parentElement.adoptedStyleSheets.includes(window.__laserStandardPlotStyles)) parentElement.adoptedStyleSheets=[...parentElement.adoptedStyleSheets,window.__laserStandardPlotStyles];
    if(!state||state.token!==data.token) {
      if(state) {state.observer.disconnect();window.__laserStandardPlotly.purge(state.plot);window.__laserStandardPlotly.purge(state.bar);parentElement.replaceChildren();}
      state=createWorkspace(parentElement,data);state.token=data.token;parentElement.__modeWorkspace=state;
      state.emit=setTriggerValue;renderRows(state,true);state.requestedAt=performance.now();state.version++;updateWorkspace(state);
    } else state.emit=setTriggerValue;
  }).catch(error=>{parentElement.appendChild(node('p','Viewer could not load: '+error.message));console.error(error);});
  return ()=>requestAnimationFrame(()=>{if(!host.isConnected && state) {state.observer.disconnect();window.__laserStandardPlotly.purge(state.plot);window.__laserStandardPlotly.purge(state.bar);}});
}
