const node = (tag, text='', className='') => {
  const element=document.createElement(tag); if(text) element.textContent=text;
  if(className) element.className=className; return element;
};
const numberText = (value, digits=3) => value==null || !Number.isFinite(Number(value))?'—':Number(value).toFixed(digits);
const plotFont = {family:'Arial, sans-serif',size:12};
const panelRows={positive:[.56,1],negative:[0,.44]};
const panelFrame=state=>{
  const width=state.plot?.clientWidth??700;
  return {height:state.view==='All 4 modes'?540:510,margin:{t:width<500?140:width<700?100:70,b:52,autoexpand:false}};
};
const traceOrder=['background','significantDown','significantUp','adduct','cluster','annotatedCluster','significantCluster','focus'];
const legendNames={background:'Not significant',significantDown:'Significantly lower (B vs A)',significantUp:'Significantly higher (B vs A)',adduct:'Other adduct candidate',
  cluster:'Cluster member',annotatedCluster:'Annotated lipid in cluster',significantCluster:'Significant cluster member',focus:'Selected / possible counterpart'};

export function modeFigure(state) {
  const {model}=state, context=model.context(state.selected), visible=model.visibleStreams(state.view);
  const phases=[...new Set(visible.map(stream=>stream.phase))];
  const traces=[], shapes=[], annotations=[];
  const visibleKeys=new Set();
  for(const key of model.hits) if(visible.some(stream=>key.startsWith(stream.name+'\u0000'))) visibleKeys.add(key);
  for(const group of [context.focus,context.cluster,context.adducts]) for(const key of group) visibleKeys.add(key);
  if(state.grey) for(const stream of visible) for(const index of stream.background) visibleKeys.add(featureKey(stream.name,stream.ids[index]));
  const frame=panelFrame(state);
  const layout={...frame,margin:{...frame.margin,l:62,r:15},
    paper_bgcolor:state.canvas,plot_bgcolor:state.canvas,font:{...plotFont,color:state.ink},
    showlegend:true,legend:{orientation:'h',x:0,y:1+(frame.margin.t-8)/(frame.height-frame.margin.t-frame.margin.b),yanchor:'top',font:{size:10}},hovermode:'closest',
    uirevision:state.view,dragmode:state.dragMode??'pan',shapes,annotations,meta:{pairs:[]}};
  phases.forEach((phase,column)=>{
    let maxRT=1,maxMZ=100;
    for(const stream of visible.filter(item=>item.phase===phase)) {
      maxMZ=Math.max(maxMZ,stream.mz[stream.mz.length-1]??100);
      for(const value of stream.rt) if(value>maxRT) maxRT=value;
    }
    const domain=phases.length===1?[0,1]:column===0?[0,.45]:[.55,1];
    const axes=[];
    for(const negative of [false,true]) {
      const index=column*2+(negative?2:1),suffix=index===1?'':String(index),xaxis='x'+suffix,yaxis='y'+suffix;
      axes.push({x:'xaxis'+suffix,y:'yaxis'+suffix});
      const buckets=new Map(traceOrder.map(role=>[role,[]]));
      for(const key of visibleKeys) {
        const point=model.point(key);if(!point||point.phase!==phase||point.negative!==negative) continue;
        buckets.get(model.style(key,context)).push(point);
      }
      traceOrder.forEach(role=>{
        const points=buckets.get(role),colour=MODE_COLOURS[role];
        traces.push({type:'scattergl',mode:'markers',xaxis,yaxis,name:legendNames[role],legendgroup:role,
          showlegend:column===0&&!negative,x:points.map(p=>p.rt),y:points.map(p=>negative?-p.mz:p.mz),
          customdata:points.map(p=>[p.key,p.stream,p.id,p.mz,p.compoundId||model.point(context.clusterOrigins?.get(p.key))?.compoundId||'unassigned',
            model.rows.get(p.key)?.adduct??'',model.rows.get(p.key)?.p_value??null,p.clusterId||model.point(context.clusterOrigins?.get(p.key))?.clusterId||'unassigned',p.lipidId?'Lipid: '+p.lipidId:'']),
          marker:{color:colour,opacity:role==='background'?.32:1,size:role==='focus'?12:8,symbol:'circle',line:{color:role==='annotatedCluster'?'#2dbd64':role==='significantCluster'?MODE_COLOURS.cluster:colour,width:role==='annotatedCluster'?3:role==='significantCluster'?2:0}},
          hovertemplate:'Compound: %{customdata[4]}<br>Cluster: %{customdata[7]}<br>%{customdata[1]} · feature %{customdata[2]}<br>'+
            '%{customdata[8]}<br>'+
            'm/z %{customdata[3]:.5f} · RT %{x:.3f} min<br>%{customdata[5]}<br>raw p %{customdata[6]:.3g}<extra>'+legendNames[role]+'</extra>'});
      });
      const zoom=state.zoomByPhase?.[phase],rt=zoom?.rt??[0,maxRT*1.02],mass=zoom?.mz??[0,maxMZ*1.04];
      const common={gridcolor:state.grid,zeroline:false,linecolor:state.grid,fixedrange:false,autorange:false};
      layout['xaxis'+suffix]={...common,anchor:negative?yaxis:'free',position:negative?undefined:.5,domain,uirevision:phase+':rt:'+(state.axisRevisions?.[phase]?.rt??0),
        showticklabels:!negative,ticks:negative?'':undefined,matches:negative?axes[0].x.replace('axis',''):undefined,
        title:{text:negative?'Retention time (min)':'',standoff:12},range:rt};
      layout['yaxis'+suffix]={...common,anchor:xaxis,uirevision:phase+':mz:'+(state.axisRevisions?.[phase]?.mz??0),title:{text:negative?'m/z (NEG)':'m/z (POS)'},
        domain:negative?panelRows.negative:panelRows.positive,range:negative?[-mass[1],-mass[0]]:mass};
      annotations.push({xref:'paper',yref:'paper',x:domain[0],y:negative?-.045:1.03,text:phase+(negative?' NEG':' POS'),
        showarrow:false,xanchor:'left',yanchor:negative?'top':'auto',font:{size:12,color:state.ink}});
    }
    layout.meta.pairs.push({phase,positive:axes[0],negative:axes[1],defaultRT:[0,maxRT*1.02],defaultMZ:[0,maxMZ*1.04]});
  });
  return {data:traces,layout};
}

export function linkedModeZoom(event, layout) {
  const updates={};
  const changed=axis=>Object.keys(event).some(key=>key.startsWith(axis+'.range')||key===axis+'.autorange');
  const range=axis=>event[axis+'.range']??[event[axis+'.range[0]']??layout[axis]?.range?.[0],event[axis+'.range[1]']??layout[axis]?.range?.[1]];
  for(const pair of layout.meta?.pairs??[]) {
    for(const dimension of ['x','y']) {
      const positive=pair.positive[dimension],negative=pair.negative[dimension];
      const source=changed(positive)?positive:changed(negative)?negative:null;if(!source) continue;
      const target=source===positive?negative:positive;
      if(event[source+'.autorange']===true) {
        updates[positive+'.range']=dimension==='x'?pair.defaultRT:pair.defaultMZ;
        updates[negative+'.range']=dimension==='x'?pair.defaultRT:[-pair.defaultMZ[1],-pair.defaultMZ[0]];
        updates[positive+'.autorange']=false;updates[negative+'.autorange']=false;
      } else {
        const values=range(source);if(values.length!==2||values.some(value=>!Number.isFinite(value))) continue;
        updates[target+'.range']=dimension==='x'?values:[-values[1],-values[0]];
        updates[target+'.autorange']=false;
      }
    }
  }
  return updates;
}

export function axisResetZoom(axis, layout) {
  for(const pair of layout.meta?.pairs??[]) for(const dimension of ['x','y']) {
    if(axis===pair.positive[dimension]||axis===pair.negative[dimension])
      return {phase:pair.phase,dimension:dimension==='x'?'rt':'mz',range:[...(dimension==='x'?pair.defaultRT:pair.defaultMZ)]};
  }
  return null;
}

export function axisFromClasses(classes) {
  for(const name of classes) {
    const match=/^([xy])(\d*)(?:tick|title)$/.exec(name);
    if(match) return match[1]+'axis'+match[2];
  }
  const subplot=classes.map(name=>/^x(\d*)y(\d*)$/.exec(name)).find(Boolean);
  if(!subplot) return null;
  if(classes.some(name=>['ewdrag','edrag','wdrag'].includes(name))) return 'xaxis'+subplot[1];
  if(classes.some(name=>['nsdrag','ndrag','sdrag'].includes(name))) return 'yaxis'+subplot[2];
  return null;
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
  const title=row?row.stream.replace('_',' ')+' · '+row.compound_id+' · '+(row.adduct||'adduct unresolved'):'Select a significant feature';
  return {data:traces,layout:{height:280,margin:{l:54,r:20,t:38,b:45},title:{text:title,x:.5,xanchor:'center',font:{size:13}},
    paper_bgcolor:state.canvas,plot_bgcolor:state.canvas,font:{...plotFont,color:state.ink},
    xaxis:{showticklabels:false,ticks:'',gridcolor:state.grid},yaxis:{title:{text:'intensity'},rangemode:'tozero',tickformat:'.2s',gridcolor:state.grid},
    barmode:'overlay',bargap:.12,showlegend:true,legend:{orientation:'h',y:-.08,x:0,font:{size:10}}}};
}

export function sampleFigures(state) {
  if(!state.selected) return [{mode:'empty',figure:sampleFigure(state)}];
  const {model}=state,compound=model.rows.get(state.selected)?.compound_id;
  const rank=new Map(model.data.adducts.map((item,index)=>[item.name,index]));
  const remaining=[...model.rows].filter(([,row])=>compound&&row.compound_id===compound)
    .sort(([a,ra],[b,rb])=>(rank.get(ra.adduct)??99)-(rank.get(rb.adduct)??99)||a.localeCompare(b)).map(([key])=>key);
  const keys=[state.selected,...[...model.context(state.selected).focus].filter(key=>key!==state.selected).sort(),...remaining];
  const seen=new Set(),figures=[],anchors=new Map();
  const selected=model.point(state.selected);if(selected) anchors.set(selected.phase,selected);
  for(const key of keys) {
    const point=state.model.point(key);if(!point||seen.has(point.stream)||!state.model.bars(key)) continue;
    const anchor=anchors.get(point.phase);if(anchor&&!model.coelutes(anchor,point)) continue;
    if(!anchor) anchors.set(point.phase,point);
    seen.add(point.stream);figures.push({mode:point.stream,key,figure:sampleFigure({...state,selected:key})});
  }
  return figures;
}

export function samplePanelFigure(state) {
  const found=new Map(sampleFigures(state).filter(item=>item.mode!=='empty').map(item=>[item.mode,item]));
  const modes=['RP_Pos','NP_Pos','RP_Neg','NP_Neg'],traces=[],annotations=[],legendGroups=new Set();
  const frame=panelFrame(state);
  const layout={...frame,margin:{...frame.margin,l:44,r:8},paper_bgcolor:state.canvas,plot_bgcolor:state.canvas,
    font:{...plotFont,color:state.ink},showlegend:true,barmode:'overlay',bargap:.12,
    legend:{orientation:'h',x:.5,xanchor:'center',y:-.035,font:{size:10}},annotations};
  modes.forEach((mode,index)=>{
    const suffix=index===0?'':String(index+1),xaxis='x'+suffix,yaxis='y'+suffix;
    const column=index%2,row=Math.floor(index/2),domain=column===0?[0,.45]:[.55,1];
    const item=found.get(mode),point=item&&state.model.point(item.key),detail=item&&state.model.rows.get(item.key);
    const header=point?'<b>'+mode.replace('_',' ').toUpperCase()+'</b><br>m/z '+numberText(point.mz,5)+' · RT '+numberText(point.rt,3)+' min · '+
      (detail?.adduct||'adduct unresolved')+'<br>'+(detail?.compound_id||'compound unassigned')+' · Feature '+point.id:
      '<b>'+mode.replace('_',' ').toUpperCase()+'</b><br>m/z — · RT — · Adduct —<br>Compound — · Feature —';
    annotations.push({xref:'paper',yref:'paper',x:(domain[0]+domain[1])/2,y:row===0?1.10:.54,
      xanchor:'center',yanchor:'top',showarrow:false,text:header,font:{size:10,color:state.ink}});
    layout['xaxis'+suffix]={domain,anchor:yaxis,showticklabels:false,ticks:'',gridcolor:state.grid,visible:!!point};
    layout['yaxis'+suffix]={domain:row===0?panelRows.positive:panelRows.negative,anchor:xaxis,title:{text:column===0?'intensity':''},
      gridcolor:state.grid,rangemode:'tozero',tickformat:'.2s',visible:!!point};
    if(item) for(const trace of item.figure.data) {
      const showlegend=!legendGroups.has(trace.name);legendGroups.add(trace.name);
      traces.push({...trace,xaxis,yaxis,legendgroup:trace.name,showlegend});
    }
    if(!point) annotations.push({xref:'paper',yref:'paper',x:(domain[0]+domain[1])/2,y:row===0?.785:.215,
      text:state.selected?'Not detected':'Select a feature',showarrow:false,font:{size:12,color:state.ink}});
  });
  return {data:traces,layout,modes:[...found.keys()]};
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
  const top=node('div','','mode-top'),main=node('div','','mode-main'),plot=node('div','','mode-plot'),side=node('aside','','mode-side'),barStack=node('div','','sample-bar-stack');
  const resets=new Map();
  for(const phase of ['RP','NP']) {const button=node('button','Reset '+phase);button.type='button';controls.appendChild(button);resets.set(phase,button);}
  main.appendChild(plot);side.appendChild(barStack);
  top.append(main,side);
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
  main.append(tableTitle,tableTools,table);root.append(controls,top,details);parent.appendChild(root);
  const state={root,model:new ModeViewerModel(data),buttons,resets,plot,barStack,barCharts:new Map(),filter,tableViewport,tableBody,tableCount,detailTitle,detailFormula,details,...evidence,
    version:0,busy:false,grey:false,zoomByPhase:{},axisRevisions:{},view:Object.keys(data.streams).some(s=>s.startsWith('RP'))?'RP POS and NEG':'NP POS and NEG',timings:[]};
  state.canvas=data.theme==='Midnight'?'#20313f':'#fff';state.ink=data.theme==='Midnight'?'#e7edf2':'#1d1d1f';state.grid=data.theme==='Midnight'?'#3b5262':'#e1e6ef';
  state.selected=null;
  state.select=(key,scroll=true)=>{if(!state.model.rows.has(key)) return;state.selected=key;renderRows(state,scroll);state.requestedAt=performance.now();state.version++;updateWorkspace(state);};
  for(const [view,button] of buttons) button.addEventListener('click',()=>{state.view=view;renderRows(state,false);state.requestedAt=performance.now();state.version++;updateWorkspace(state);});
  for(const [phase,button] of resets) button.addEventListener('click',()=>{
    const pair=state.plot.layout.meta?.pairs.find(pair=>pair.phase===phase);if(!pair) return;
    const revision=state.axisRevisions[phase]??{rt:0,mz:0};state.axisRevisions[phase]={rt:revision.rt+1,mz:revision.mz+1};
    state.zoomByPhase[phase]={rt:[...pair.defaultRT],mz:[...pair.defaultMZ]};state.requestedAt=performance.now();state.version++;updateWorkspace(state);
  });
  grey.addEventListener('change',()=>{state.grey=grey.checked;state.requestedAt=performance.now();state.version++;updateWorkspace(state);});
  filter.addEventListener('input',()=>{tableViewport.scrollTop=0;renderRows(state,false);});
  tableViewport.addEventListener('scroll',()=>renderRows(state,false));
  review.addEventListener('click',()=>{if(state.selected) state.emit?.('review',state.selected.split('\u0000'));});
  state.observer=new ResizeObserver(()=>{if(state.plot._fullLayout) {
    const width=state.plot.clientWidth;
    if(width!==state.plotWidth) {state.plotWidth=width;state.version++;updateWorkspace(state);}
    window.__laserStandardPlotly.Plots.resize(state.plot);for(const chart of state.barCharts.values()) window.__laserStandardPlotly.Plots.resize(chart);
  }});
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
      for(const [phase,button] of state.resets) button.hidden=!state.model.visibleStreams(state.view).some(stream=>stream.phase===phase);
      renderEvidence(state);
      const figure=modeFigure(state),panel=samplePanelFigure(state),bars=[{mode:'panel',figure:panel}],Plotly=window.__laserStandardPlotly;
      const modes=new Set(['panel']);
      for(const [mode,chart] of state.barCharts) if(!modes.has(mode)) {Plotly.purge(chart);chart.remove();state.barCharts.delete(mode);}
      const updates=[Plotly.react(state.plot,figure.data,figure.layout,{displaylogo:false,scrollZoom:true,responsive:true,doubleClick:false,modeBarButtonsToRemove:['zoomIn2d','zoomOut2d','autoScale2d','resetScale2d','select2d','lasso2d']})];
      for(const item of bars) {
        let chart=state.barCharts.get(item.mode);
        if(!chart) {chart=node('div','','sample-panel');chart.dataset.mode=item.mode;state.barCharts.set(item.mode,chart);}
        state.barStack.appendChild(chart);
        updates.push(Plotly.react(chart,item.figure.data,item.figure.layout,{displayModeBar:false,responsive:true}));
      }
      await Promise.all(updates);state.root.dataset.zoomRanges=JSON.stringify(state.zoomByPhase);state.root.dataset.barModes=JSON.stringify(panel.modes);
      if(!state.bound) {
        const resetAxisFromPointer=event=>{
          const classes=event.composedPath().flatMap(element=>(element.getAttribute?.('class')??'').split(/\s+/));
          const axis=axisFromClasses(classes),reset=axisResetZoom(axis,state.plot.layout);if(!reset) return;
          if(event.type==='mousedown') {
            const previous=state.axisClick,now=performance.now();
            state.axisClick={axis,time:now,x:event.clientX,y:event.clientY};
            if(event.detail<2&&!(previous?.axis===axis&&now-previous.time<400&&Math.hypot(event.clientX-previous.x,event.clientY-previous.y)<8)) return;
          }
          event.preventDefault();event.stopImmediatePropagation();
          const pair=state.plot.layout.meta.pairs.find(pair=>pair.phase===reset.phase);
          const current=state.zoomByPhase[reset.phase]??{rt:[...state.plot.layout[pair.positive.x].range],mz:[...state.plot.layout[pair.positive.y].range]};
          state.zoomByPhase[reset.phase]={...current,[reset.dimension]:reset.range};
          const revision=state.axisRevisions[reset.phase]??{rt:0,mz:0};state.axisRevisions[reset.phase]={...revision,[reset.dimension]:revision[reset.dimension]+1};
          state.requestedAt=performance.now();state.version++;updateWorkspace(state);
        };
        // Plotly's axis drag layer consumes clicks; intercept the second
        // mouse-down before its drag handler, with dblclick as a fallback.
        state.plot.addEventListener('mousedown',resetAxisFromPointer,true);
        state.plot.addEventListener('dblclick',resetAxisFromPointer,true);
        state.plot.on('plotly_click',event=>{const key=event.points?.[0]?.customdata?.[0];if(key&&state.model.rows.has(key)) state.select(key,true);});
        state.plot.on('plotly_relayout',async event=>{
          if(['pan','zoom'].includes(event.dragmode)) state.dragMode=event.dragmode;
          if(state.syncingZoom) return;
          const updates=linkedModeZoom(event,state.plot.layout);
          state.syncingZoom=true;
          try {
            if(Object.keys(updates).length) await Plotly.relayout(state.plot,updates);
            for(const pair of state.plot.layout.meta?.pairs??[]) state.zoomByPhase[pair.phase]={
              rt:[...state.plot.layout[pair.positive.x].range],mz:[...state.plot.layout[pair.positive.y].range]};
            state.root.dataset.zoomRanges=JSON.stringify(state.zoomByPhase);
          } finally {state.syncingZoom=false;}
        });state.bound=true;
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
      if(state) {state.observer.disconnect();window.__laserStandardPlotly.purge(state.plot);for(const chart of state.barCharts?.values()??[]) window.__laserStandardPlotly.purge(chart);parentElement.replaceChildren();}
      state=createWorkspace(parentElement,data);state.token=data.token;parentElement.__modeWorkspace=state;
      state.emit=setTriggerValue;renderRows(state,true);state.requestedAt=performance.now();state.version++;updateWorkspace(state);
    } else state.emit=setTriggerValue;
  }).catch(error=>{parentElement.appendChild(node('p','Viewer could not load: '+error.message));console.error(error);});
  return ()=>requestAnimationFrame(()=>{if(!host.isConnected && state) {state.observer.disconnect();window.__laserStandardPlotly.purge(state.plot);for(const chart of state.barCharts.values()) window.__laserStandardPlotly.purge(chart);}});
}
