export const MODE_COLOURS = {
  significantUp:'#d62728', significantDown:'#2677c9', focus:'#00e51b', cluster:'#8b5cf6', significantCluster:'#f28c28', adduct:'#f2c230', background:'#aeb6c1'
};
export const featureKey = (stream, id) => stream + '\u0000' + String(id);
const decode64 = value => {
  const bytes=Uint8Array.from(atob(value), letter=>letter.charCodeAt(0));
  return new Float64Array(bytes.buffer);
};
const lowerBound = (values, target) => {
  let lo=0, hi=values.length;
  while(lo<hi) {const mid=(lo+hi)>>>1; if(values[mid]<target) lo=mid+1; else hi=mid;}
  return lo;
};

export class ModeViewerModel {
  constructor(data) {
    this.data=data; this.streams=new Map(); this.points=new Map(); this.rows=new Map();
    this.hits=new Set(); this.links=new Map(); this.adjacency=new Map(); this.queryCache=new Map();
    this.componentCache=new Map(); this.contextCache=new Map();
    this.clusterCompounds=new Map(); this.clusterIds=new Map(); this.namedClusters=new Map();
    for(const [name, source] of Object.entries(data.streams)) {
      const stream={...source, name, mz:decode64(source.mz), rt:decode64(source.rt), levels:decode64(source.levels)};
      stream.sampleIndex=new Map(source.sampleIds.map((id,index)=>[String(id),index]));
      stream.negative=name.toLowerCase().endsWith('_neg'); stream.phase=name.split('_')[0];
      this.streams.set(name,stream);
      source.ids.forEach((id,index)=>this.points.set(featureKey(name,id),{stream,index,key:featureKey(name,id)}));
    }
    for(const row of data.statistics) {
      const key=featureKey(row.stream,row.feature_id); this.rows.set(key,row);
      if(row.raw_p_significant) this.hits.add(key);
    }
    for(const row of data.clusterMembers??[]) {
      const key=featureKey(row.stream,row.feature_id);this.clusterIds.set(key,row.cluster_id);if(row.compound_id) this.clusterCompounds.set(key,row.compound_id);
      if(!this.namedClusters.has(row.cluster_id)) this.namedClusters.set(row.cluster_id,new Set());
      this.namedClusters.get(row.cluster_id).add(key);
    }
    for(const item of data.links) {
      const [ss,si,ts,ti,sa,ta,ppm,rt,compared,neutral]=item;
      const source=featureKey(ss,si), target=featureKey(ts,ti);
      const forward={source,target,sourceAdduct:sa,targetAdduct:ta,ppm,rt,compared,neutral};
      const reverse={source:target,target:source,sourceAdduct:ta,targetAdduct:sa,ppm:-ppm,rt:-rt,compared,neutral};
      for(const link of [forward,reverse]) {
        if(!this.links.has(link.source)) this.links.set(link.source,[]);
        this.links.get(link.source).push(link);
      }
    }
    for(const [stream,source,target] of data.clusters) {
      const a=featureKey(stream,source), b=featureKey(stream,target);
      if(this.rows.get(a)?.compound_id && this.rows.get(a)?.compound_id===this.rows.get(b)?.compound_id) continue;
      if(!this.adjacency.has(a)) this.adjacency.set(a,new Set());
      if(!this.adjacency.has(b)) this.adjacency.set(b,new Set());
      this.adjacency.get(a).add(b); this.adjacency.get(b).add(a);
    }
  }
  point(key) {
    const point=this.points.get(key); if(!point) return null;
    return {key,stream:point.stream.name,id:point.stream.ids[point.index],
      mz:point.stream.mz[point.index],rt:point.stream.rt[point.index],negative:point.stream.negative,phase:point.stream.phase,lipidId:point.stream.lipidIds?.[point.stream.ids[point.index]]??'',
      compoundId:this.rows.get(key)?.compound_id??this.clusterCompounds.get(key)??'',clusterId:this.rows.get(key)?.cluster_id??this.clusterIds.get(key)??''};
  }
  matches(key) {
    if(this.queryCache.has(key)) return this.queryCache.get(key);
    const saved=this.links.get(key)??[];
    // Saved searches cover raw-p hits. Non-significant cluster neighbours need
    // a small display-only mass/RT lookup against already-loaded coordinates.
    if(this.hits.has(key)) {this.queryCache.set(key,saved); return saved;}
    const point=this.point(key); if(!point) return saved;
    const row=this.rows.get(key), observedCharge=row?.isotope_charge_hypothesis;
    const sources=this.data.adducts.filter(a=>a.polarity===(point.negative?'neg':'pos') && (!observedCharge || a.charge===observedCharge));
    const ppm=this.data.criteria.cross_mode_ppm_tolerance??3;
    const rtTolerance=this.data.criteria.opposite_polarity_rt_tolerance_min??.1;
    const matches=[...saved];
    for(const source of sources) {
      const neutral=point.mz*source.charge-source.offset; if(neutral<=0) continue;
      for(const stream of this.streams.values()) {
        const samePhase=point.phase===stream.phase;
        for(const target of this.data.adducts) {
          if(target.polarity!==(stream.negative?'neg':'pos')) continue;
          if(stream.name===point.stream && target.name===source.name) continue;
          const expected=(neutral+target.offset)/target.charge; if(expected<=0) continue;
          const error=expected*ppm*1e-6;
          for(let index=lowerBound(stream.mz,expected-error);index<stream.mz.length && stream.mz[index]<=expected+error;index++) {
            const targetKey=featureKey(stream.name,stream.ids[index]); if(targetKey===key) continue;
            const targetCharge=this.rows.get(targetKey)?.isotope_charge_hypothesis;
            if(targetCharge && targetCharge!==target.charge) continue;
            const rt=(stream.rt[index]-point.rt)*60;
            if(samePhase && Math.abs(rt)>rtTolerance*60) continue;
            matches.push({source:key,target:targetKey,sourceAdduct:source.name,targetAdduct:target.name,
              ppm:(stream.mz[index]-expected)/expected*1e6,rt,compared:samePhase,neutral});
          }
        }
      }
    }
    this.queryCache.set(key,matches); return matches;
  }
  cluster(key) {
    if(this.clusterIds.has(key)) return this.namedClusters.get(this.clusterIds.get(key));
    if(this.componentCache.has(key)) return this.componentCache.get(key);
    const members=new Set(), stack=[key];
    while(stack.length) {
      const current=stack.pop(); if(members.has(current)) continue;
      members.add(current);
      for(const next of this.adjacency.get(current)??[]) if(!members.has(next)) stack.push(next);
    }
    for(const member of members) this.componentCache.set(member,members);
    return members;
  }
  context(key) {
    if(this.contextCache.has(key)) return this.contextCache.get(key);
    if(!key) return {key:null,focus:new Set(),cluster:new Set(),adducts:new Set(),matches:[]};
    const focus=new Set([key]), cluster=new Set(), adducts=new Set(), clusterOrigins=new Map();
    const selected=this.point(key), matches=this.matches(key), byMode=new Map(), phaseAnchors=new Map();
    if(selected) phaseAnchors.set(selected.phase,selected);
    const compound=this.rows.get(key)?.compound_id;
    const rank=new Map(this.data.adducts.map((item,index)=>[item.name,index]));
    for(const link of matches) {
      const target=this.point(link.target); if(!target) continue;
      if(compound && this.rows.get(link.target)?.compound_id!==compound) continue;
      if(selected?.phase===target.phase&&!this.coelutes(selected,target)) continue;
      if(target.stream===selected?.stream) continue;
      if(!byMode.has(target.stream)) byMode.set(target.stream,[]);
      byMode.get(target.stream).push(link);
    }
    // One representative ion per other mode is red. Further ion/adduct
    // candidates stay yellow; all original alternatives remain in evidence.
    for(const [,links] of [...byMode.entries()].sort(([a],[b])=>a.localeCompare(b))) {
      links.sort((a,b)=>(rank.get(a.targetAdduct)??99)-(rank.get(b.targetAdduct)??99)||Math.abs(a.ppm)-Math.abs(b.ppm));
      const chosen=links.find(link=>{const target=this.point(link.target),anchor=phaseAnchors.get(target.phase);return !anchor||this.coelutes(anchor,target);});
      if(chosen) {const target=this.point(chosen.target);focus.add(chosen.target);if(!phaseAnchors.has(target.phase)) phaseAnchors.set(target.phase,target);}
    }
    for(const link of matches) {
      const target=this.point(link.target);if(!target||(compound&&this.rows.get(link.target)?.compound_id!==compound)) continue;
      const anchor=phaseAnchors.get(target.phase);if(anchor&&this.coelutes(anchor,target)) adducts.add(link.target);
    }
    const localCluster=target=>{const point=this.point(target),anchor=point&&phaseAnchors.get(point.phase);return point&&(!anchor||Math.abs(point.rt-anchor.rt)<=(this.data.criteria.ch2_rt_tolerance_min??.5)+1e-12);};
    for(const root of this.hits.has(key)?[...focus]:[]) {
      // Selection highlights direct saved neighbours, not the full transitive
      // component used for cluster numbering. Never grow neighbours of neighbours.
      for(const member of this.adjacency.get(root)??[]) {
        if(focus.has(member)) continue;
        if(compound && this.rows.get(member)?.compound_id===compound) continue;
        if(!localCluster(member)) continue;
        cluster.add(member);clusterOrigins.set(member,member);
        const memberCompound=this.rows.get(member)?.compound_id;
        if(memberCompound) {
          for(const [target,row] of this.rows) if(row.compound_id===memberCompound && !focus.has(target)&&localCluster(target)&&
            this.coelutes(this.point(member),this.point(target))) {cluster.add(target);clusterOrigins.set(target,member);}
        } else {
          const representatives=new Map(),memberPoint=this.point(member);
          for(const link of this.matches(member)) {
            const target=this.point(link.target);if(!target||target.stream===memberPoint?.stream||focus.has(link.target)) continue;
            if(!localCluster(link.target)||!this.coelutes(memberPoint,target)) continue;
            const prior=representatives.get(target.stream);
            if(!prior || (rank.get(link.targetAdduct)??99)<(rank.get(prior.targetAdduct)??99) ||
              ((rank.get(link.targetAdduct)??99)===(rank.get(prior.targetAdduct)??99) && Math.abs(link.ppm)<Math.abs(prior.ppm))) representatives.set(target.stream,link);
          }
          for(const link of representatives.values()) {cluster.add(link.target);clusterOrigins.set(link.target,member);}
        }
      }
    }
    for(const member of focus) {cluster.delete(member);adducts.delete(member);}
    const result={key,focus,cluster,adducts,matches,clusterOrigins};
    if(this.contextCache.size>128) this.contextCache.delete(this.contextCache.keys().next().value);
    this.contextCache.set(key,result); return result;
  }
  coelutes(a,b) {
    if(!a||!b) return false;
    if(a.phase!==b.phase) return true;
    const tolerance=a.stream===b.stream?(this.data.criteria.same_mode_adduct_rt_tolerance_min??this.data.criteria.isotope_rt_tolerance_min??.05):
      (this.data.criteria.opposite_polarity_rt_tolerance_min??.2);
    return Math.abs(a.rt-b.rt)<=tolerance+1e-12;
  }
  style(key, context) {
    if(context.focus.has(key)) return 'focus';
    if(context.cluster.has(key)) return this.hits.has(key)?'significantCluster':'cluster';
    if(context.adducts.has(key)) return 'adduct';
    return this.hits.has(key)?(Number(this.rows.get(key)?.log2_fold_change_b_vs_a)>0?'significantUp':'significantDown'):'background';
  }
  visibleStreams(view) {
    return [...this.streams.values()].filter(stream=>view==='All 4 modes'||stream.phase===view.split(' ')[0]);
  }
  tableRows(view, search='') {
    const modes=new Set(this.visibleStreams(view).map(stream=>stream.name)), query=search.toLowerCase();
    return [...this.rows.entries()].filter(([,row])=>modes.has(row.stream) &&
      (!query||[row.compound_id,row.stream,row.feature_id,row.adduct,row.mz].join(' ').toLowerCase().includes(query)));
  }
  bars(key) {
    const point=this.point(key); if(!point) return null;
    const stream=this.streams.get(point.stream), index=stream.sampleIndex.get(String(point.id));
    if(index===undefined) return null;
    const start=index*stream.samples.length;
    return {stream,samples:stream.samples,values:stream.levels.subarray(start,start+stream.samples.length)};
  }
}
