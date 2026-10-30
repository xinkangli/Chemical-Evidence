"""Frozen multi-source replication screen. Never stop at a favorable source."""
from pathlib import Path
import io,json,pickle,zipfile,hashlib,time,datetime,sys
import numpy as np,pandas as pd
import chemical_evidence_v1 as ce
import evidence_review_agent_v2 as m
from run_external_chemical_gate_v1 import threshold,accept
R=Path(__file__).resolve().parents[1]
REPAIR='--edge-domain-audit' in sys.argv
O=R/('reports/public_chemical_batch_edge_audit_v1' if REPAIR else 'reports/public_chemical_batch_v1')
MAP={'PE':'polyethylene','HDPE':'polyethylene','LDPE':'polyethylene','PP':'polypropylene','PS':'polystyrene','PET':'polyethylene_terephthalate'}

def dump(name,obj):
 (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def collect():
 panels={};ledger=[]
 def add(panel,cid,group,label,x,t,eligible=True,**extra):
  x=np.asarray(x,float);t=np.asarray(t,float);order=np.argsort(x);x=x[order];t=t[order]
  rawhash=hashlib.sha256(np.column_stack([x,t]).tobytes()).hexdigest()
  edge_removed=0
  if REPAIR:
   pos=np.flatnonzero(t>0);lo,hi=pos[[0,-1]]
   assert (t[lo:hi+1]>0).all() and (t[:lo]==0).all() and (t[hi+1:]==0).all()
   edge_removed=int(lo+len(t)-hi-1);x=x[lo:hi+1];t=t[lo:hi+1]
   assert x[0]<=702 and x[-1]>=1800
  valid=bool(np.isfinite(x).all() and np.isfinite(t).all() and (t>0).all() and (np.diff(x)>0).all())
  y=-np.log10(t) if valid else None
  row=dict(panel=panel,case_id=cid,group=group,reported_label=label,truth=MAP.get(label,'unknown'),known=label in MAP,eligible=eligible,valid=valid,raw_sha256=rawhash,unusable_edge_points=edge_removed,**extra)
  panels.setdefault(panel,[]).append((row,x,y));ledger.append(row)
 d=pd.read_parquet(R/'reports/polymer_supplement_v1/records.parquet')
 for source in ['flopp','blop']:
  dd=d[d.source.eq(source)]
  if source=='flopp':dd=dd[dd.subset.eq('FLOPP-e')]
  else:dd=dd.sort_values('archive_file').drop_duplicates('vector_sha256')
  with zipfile.ZipFile(R/('data/raw/rochman_'+source+'/library.zip')) as z:
   for r in dd.itertuples():
    arr=np.loadtxt(io.BytesIO(z.read(r.archive_file)),delimiter=',');label=str(r.reported_label)
    eligible=source=='flopp' or label in ['Polylactic Acid (PLA)','Crystalized Polylactic Acid (CPLA)','Polyhydroxyalkanoate (PHA)']
    add('FLOPP-e' if source=='flopp' else 'BLoP',r.record_id,r.conservative_product_group,label,arr[:,0],arr[:,1],eligible,
     inherited_FLOPP=bool(r.source_declares_FLOPP_overlap) if source=='blop' else False,label_status='author_library_assignment' if source=='flopp' else 'explicit_polymer' if eligible else 'brand_or_feedstock_ambiguous')
 if REPAIR:
  pd.DataFrame(ledger).to_csv(O/'intake.csv',index=False)
  return panels
 for f in sorted((R/'data/raw/zenodo_21284853').glob('*.txt')):
  head=f.read_text().splitlines();assert '%T' in head[1]
  arr=np.loadtxt(f,skiprows=2);add('PLLA-TAP',f.name,f.stem,'PLLA formulation',arr[:,0],arr[:,1],label_status='prepared_PLA_formulation_not_independent_replicates',inherited_FLOPP=False)
 with zipfile.ZipFile(R/'data/raw/ftir_plastics_10736650/FTIR-PLASTIC.zip') as z:
  for name in sorted(z.namelist()):
   if not name.endswith('.zip'):continue
   label=Path(name).stem.split('_')[0];resolution=Path(name).stem.split('_')[1]
   with zipfile.ZipFile(io.BytesIO(z.read(name))) as zz:
    for member in sorted(zz.namelist()):
     if not member.lower().endswith('.csv'):continue
     s=zz.read(member).decode('cp1252');lines=s.splitlines();assert any('ATR' in l for l in lines[:15]) and any('Transmittance' in l for l in lines[:15])
     arr=np.loadtxt(io.StringIO(s),delimiter=',',skiprows=15)
     add('FTIR-Plastics-'+resolution,name+':'+member,label,label,arr[:,0],arr[:,1],label_status='augmented_row_not_independent_specimen',inherited_FLOPP=False)
 pd.DataFrame(ledger).to_csv(O/'intake.csv',index=False)
 dump('intake_summary.json',{p:{'records':len(rs),'eligible':sum(r[0]['eligible'] for r in rs),'invalid_signal':sum(not r[0]['valid'] for r in rs)} for p,rs in panels.items()})
 return panels

def main():
 start=time.perf_counter(); assert not (O/'metrics.csv').exists()
 freeze=json.loads((O/'freeze.json').read_text());assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in freeze['hashes'].items())
 panels=collect();print('INTAKE',[(p,len(rs)) for p,rs in panels.items()],flush=True)
 base=np.load(R/'reports/grouped_evidence_v1/public_reference_inputs.npz');_,data=pickle.load((R/'reports/evidence_review_agent_v2/prepared.pkl').open('rb'))
 models=pickle.load((R/'reports/calibrated_evidence_fusion_v1/models.pkl').open('rb'))['ATR'];classes=np.array(sorted(set(base['labels'])))
 src=data['paired']['ATR'];ext=data['external'];params=json.loads((R/'reports/external_chemical_gate_v1/thresholds.json').read_text())
 metrics=[];familyrows=[];audit=[]
 for panel,records in panels.items():
  t0=time.perf_counter();meta=pd.DataFrame([r[0] for r in records]);valid=meta.valid.to_numpy(bool)
  q=m.norm(np.array([np.interp(base['axis'],x,y) if row['valid'] else np.zeros_like(base['axis']) for row,x,y in records]))
  a=m.class5(q,base['x'],base['labels'],classes);b=m.class5(q,src['x'],src['labels'],classes);sim=q@m.norm(ext['x']).T
  c=np.array([[np.sort(ss[(ext['labels']==cl)&(ss<.999999)])[-min(5,int(((ext['labels']==cl)&(ss<.999999)).sum())):].mean() for cl in classes] for ss in sim]);outputs={}
  for name,sc in [('initial_retrieval',a),('matched_retrieval',b),('fixed_fusion',(b+c)/2)]:
   ix=np.argsort(-sc,axis=1,kind='stable');v=np.take_along_axis(sc,ix[:,:2],axis=1);outputs[name]=(classes[ix[:,0]],.25*(v[:,0]+1)+.25*(v[:,0]-v[:,1]))
  for name,model in models.items():
   inp=np.concatenate([a,b,c],axis=1) if name=='learned_fusion' else q
   sc=model.decision_function(inp) if name.startswith('svc') else model.predict_proba(inp);outputs[name]=(model.classes_[sc.argmax(axis=1)],sc.max(axis=1))
  bands=[];support=[]
  for i,(row,x,y) in enumerate(records):
   if not row['valid']:support.append(-np.inf);continue
   pk=ce.check_spectrum(x,y,'ATR','absorbance',panel+':'+str(i));bb=[z for z in pk['bands'] if z['family_hypothesis']==outputs['fixed_fusion'][0][i]]
   support.append(len({z['peak_observation_id'] for z in bb if z['status']=='local_feature_detected'})/len(bb));bands.extend({'case_id':row['case_id'],**z} for z in pk['bands'])
  outputs['chemical_support']=(outputs['fixed_fusion'][0],np.array(support));pd.DataFrame(bands).to_parquet(O/(panel+'_bands.parquet'),index=False)
  # A group is a product where known. Augmented panels explicitly use row-descriptive weighting.
  meta['weight']=1. if panel.startswith('FTIR-Plastics') else 1/meta.groupby('group').case_id.transform('size')
  near=np.maximum.reduce([(q@m.norm(ref).T).max(axis=1) for ref in [base['x'],src['x'],ext['x']]])
  meta['reference_max_cosine']=near;meta['reference_near_duplicate']=near>=.999999
  scored=[]
  for name,(pred,v) in outputs.items():
   v=v.copy();v[~valid]=-np.inf;g=meta.copy();g['method']=name;g['prediction']=pred;g['score']=v;scored.append(g)
   use=meta.eligible.to_numpy(bool);k=meta.known.to_numpy(bool);w=meta.weight.to_numpy();err=pred!=meta.truth.to_numpy()
   settings=[('frozen_textile_threshold',None,params[name]['threshold'],params[name]['tie_acceptance_probability'])]
   if (use&k).any() and (use&~k).any():
    th,p,ok=threshold(v[use&k],w[use&k],.9)
    if ok:settings.append(('retrospective_matched',.9,th,p))
   for mode,target,th,p in settings:
    ac=accept(v,th,p);ks=use&k;us=use&~k;den=np.sum(w[ks]*ac[ks])
    result=dict(panel=panel,method=name,evaluation=mode,target=target,known_rows=int(ks.sum()),unknown_rows=int(us.sum()),known_acceptance=float(np.average(ac[ks],weights=w[ks])) if ks.any() else None,unknown_acceptance=float(np.average(ac[us],weights=w[us])) if us.any() else None,accepted_known_error=float(np.sum(w[ks]*ac[ks]*err[ks])/den) if den else None,threshold=float(th),tie_probability=float(p))
    metrics.append(result)
    for lab in meta.reported_label.unique():
     mask=use&meta.reported_label.eq(lab).to_numpy()
     if mask.any():familyrows.append(dict(panel=panel,method=name,evaluation=mode,label=lab,rows=int(mask.sum()),acceptance=float(np.average(ac[mask],weights=w[mask]))))
   if panel=='BLoP':
    # Same locked thresholds, inherited references separate from noninherited materials.
    for inherited in [False,True]:
     mask=use&~k&meta.inherited_FLOPP.eq(inherited).to_numpy();ac=accept(v,params[name]['threshold'],params[name]['tie_acceptance_probability'])
     if mask.any():familyrows.append(dict(panel=panel,method=name,evaluation='inheritance_stratum',label=str(inherited),rows=int(mask.sum()),acceptance=float(np.average(ac[mask],weights=w[mask]))))
  pd.concat(scored,ignore_index=True).to_parquet(O/(panel+'_scores.parquet'),index=False)
  audit.append(dict(panel=panel,rows=len(meta),eligible=int(meta.eligible.sum()),invalid=int((~valid).sum()),reference_near_duplicates=int((near>=.999999).sum()),seconds=time.perf_counter()-t0))
  print('PANEL_COMPLETE',panel,round(time.perf_counter()-t0,1),flush=True)
 pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.DataFrame(familyrows).to_csv(O/'family_metrics.csv',index=False)
 assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in freeze['hashes'].items())
 dump('verification.json',{'status':'passed','frozen_inputs_unchanged':True,'panels':audit,'seconds':time.perf_counter()-start,'all_admitted_panels_completed':True,'new_models_trained':0})
 print(pd.DataFrame(metrics).query('method in ["fixed_fusion","chemical_support"]').to_string(index=False),flush=True)

if __name__=='__main__':main()


