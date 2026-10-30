"""Single frozen external-source evaluation, no fit or parameter search."""
from pathlib import Path
import json,hashlib,pickle,datetime,time
import numpy as np
import pandas as pd
import chemical_evidence_v1 as ce
import evidence_review_agent_v2 as m
R=Path(__file__).resolve().parents[1];O=R/'reports/external_chemical_gate_v1';RAW=R/'data/raw/textile_rx3fjgz96x_v1'

def dump(name,obj):
 (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def threshold(v,w,target=.9):
 good=np.isfinite(v); attainable=w[good].sum()/w.sum()
 if attainable<target:return (-float('inf'),1.,False)
 order=np.argsort(-v,kind='stable');cum=np.cumsum(w[order]); t=float(v[order[np.searchsorted(cum,target*w.sum(),side='left')]])
 p=float((target*w.sum()-w[v>t].sum())/w[v==t].sum())
 assert -1e-8<=p<=1+1e-8
 return t,float(np.clip(p,0,1)),True

def accept(v,t,p):
 return np.where(np.isfinite(v),(v>t).astype(float)+(v==t)*p,0.)

def main():
 start=time.perf_counter(); freeze=json.loads((O/'freeze.json').read_text())
 assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in freeze['hashes'].items())
 assert not (O/'scores.csv').exists(),'No outcome-based rerun'
 raw=pd.read_csv(RAW/'02_feature_matrix_raw_T.csv',encoding='cp1252');meta=pd.read_csv(RAW/'06_metadata.csv',encoding='cp1252')
 assert meta.Sample_ID.is_unique and raw.Spectrum_ID.is_unique
 assert set(raw.Sample_ID)==set(meta.Sample_ID)
 discrepancies=[]
 for field in ['Type','Subtype']:
  mapped=raw.Sample_ID.map(meta.set_index('Sample_ID')[field])
  for i in np.flatnonzero(raw[field].ne(mapped)):
   discrepancies.append(dict(Spectrum_ID=int(raw.iloc[i].Spectrum_ID),Sample_ID=raw.iloc[i].Sample_ID,field=field,raw=raw.iloc[i][field],metadata=mapped.iloc[i]))
  if field=='Type':assert raw[field].str.lower().equals(mapped.str.lower())
  else:
   for i in np.flatnonzero(raw[field].ne(mapped)):
    assert raw.iloc[i].Sample_ID=='MTF_NYL_103' and raw.iloc[i][field].strip()=='Unspecified PA Copolymer' and mapped.iloc[i].strip()=='Unspecified PA Type'
 dump('metadata_discrepancies.json',dict(records=discrepancies,resolution='Case-only Type differences normalized; MTF_NYL_103 subtype wording differs but both describe polyamide outside fixed dictionary. Use dedicated author metadata; do not infer a PA subtype. Detected before any scoring.'))
 labelmap={'PE':'polyethylene','PP':'polypropylene','PS':'polystyrene','PET':'polyethylene_terephthalate'}
 meta['truth']=meta.Subtype.map(labelmap).fillna('unknown')
 meta['known']=meta.Subtype.isin(labelmap)
 meta['partition']='evaluation'
 for fam,g in meta[meta.known].groupby('truth'):
  ids=sorted(g.Sample_ID,key=lambda s:hashlib.sha256(('AS-external-v1|'+s).encode()).hexdigest())
  meta.loc[meta.Sample_ID.isin(ids[:len(ids)//2]),'partition']='calibration'
 meta.to_csv(O/'sample_split.csv',index=False)
 assert not set(meta[meta.partition.eq('calibration')].Sample_ID)&set(meta[meta.partition.eq('evaluation')].Sample_ID)
 dump('intake_before_scores.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  schema_clarification='Official table uses Sample_ID, not Source_ID stated on landing page; physical-sample grouping unchanged.',
  transmittance='Author pipeline explicitly uses percent T. Values above 100 are retained as negative absorbance baseline; no clipping; feature checker receives the directly converted absorbance.',
  rows=len(raw),samples=len(meta),known_samples=int(meta.known.sum()),unknown_samples=int((~meta.known).sum()),calibration_samples=int(meta.partition.eq('calibration').sum()),
  split_sha256=hashlib.sha256((O/'sample_split.csv').read_bytes()).hexdigest(),
  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.iterdir() if p.is_file()},
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 # Labels and split are not passed to any numerical scoring method.
 cols=raw.columns[6:];axis=np.array(cols,float);order=np.argsort(axis);axis=axis[order]
 T=raw[cols].to_numpy(float)[:,order];valid=np.isfinite(T).all(axis=1)&(T>0).all(axis=1)
 assert valid.all(),'Retain invalid records and implement explicit rejection before proceeding'
 A=-np.log10(T/100.)
 _,data=pickle.load((R/'reports/evidence_review_agent_v2/prepared.pkl').open('rb'))
 base=np.load(R/'reports/grouped_evidence_v1/public_reference_inputs.npz')
 classes=np.array(sorted(set(base['labels'])));q=m.norm(np.array([np.interp(base['axis'],axis,y) for y in A]))
 a=m.class5(q,base['x'],base['labels'],classes);src=data['paired']['ATR'];b=m.class5(q,src['x'],src['labels'],classes)
 ext=data['external'];sim=m.norm(q)@m.norm(ext['x']).T;c=[]
 for ss in sim:
  c.append([np.sort(ss[(ext['labels']==cl)&(ss<.999999)])[-min(5,int(((ext['labels']==cl)&(ss<.999999)).sum())):].mean() for cl in classes])
 c=np.array(c); assert np.isfinite(c).all()
 outputs={}
 for name,sc in [('initial_retrieval',a),('matched_retrieval',b),('fixed_fusion',(b+c)/2)]:
  ix=np.argsort(-sc,axis=1,kind='stable');v=np.take_along_axis(sc,ix[:,:2],axis=1)
  outputs[name]=(classes[ix[:,0]],.25*(v[:,0]+1)+.25*(v[:,0]-v[:,1]))
 models=pickle.load((R/'reports/calibrated_evidence_fusion_v1/models.pkl').open('rb'))['ATR']
 for name,model in models.items():
  x=np.concatenate([a,b,c],axis=1) if name=='learned_fusion' else q
  sc=model.decision_function(x) if name.startswith('svc') else model.predict_proba(x)
  outputs[name]=(model.classes_[sc.argmax(axis=1)],sc.max(axis=1))
 bandrows=[];support=[]
 for i,y in enumerate(A):
  packet=ce.check_spectrum(axis,y,'ATR','absorbance','textile-'+str(raw.iloc[i].Spectrum_ID))
  candidate=outputs['fixed_fusion'][0][i]
  bands=[z for z in packet['bands'] if z['family_hypothesis']==candidate]
  peaks={z['peak_observation_id'] for z in bands if z['status']=='local_feature_detected'}
  support.append(len(peaks)/len(bands))
  bandrows.extend({'Spectrum_ID':int(raw.iloc[i].Spectrum_ID),**z} for z in packet['bands'])
 outputs['chemical_support']=(outputs['fixed_fusion'][0],np.array(support))
 pd.DataFrame(bandrows).to_parquet(O/'band_evidence.parquet',index=False)
 overlap=[]
 for name,refs in [('initial',base['x']),('matched',src['x']),('external',ext['x'])]:
  cs=q@m.norm(refs).T
  overlap.extend(dict(Spectrum_ID=int(s),pool=name,max_cosine=float(v),reference_index=int(j),near_duplicate=bool(v>=.999999)) for s,v,j in zip(raw.Spectrum_ID,cs.max(axis=1),cs.argmax(axis=1)))
 pd.DataFrame(overlap).to_csv(O/'reference_overlap.csv',index=False)
 # Include model fitting/calibration input vectors, not only retrieval references.
 fit=np.load(R/'reports/paired_calibration_v1/inputs.npz')['x']; fd=pd.read_parquet(R/'reports/paired_calibration_v1/aggregate_records.parquet')
 fit=fit[fd.partition.isin(['train','calibration']) & fd['mode'].eq('ATR')]
 fitcs=q@m.norm(fit).T
 dump('overlap_audit.json',dict(reference_near_duplicates=sum(x['near_duplicate'] for x in overlap),model_fit_near_duplicates=int((fitcs.max(axis=1)>=.999999).sum()),model_fit_max_cosine=float(fitcs.max()),
  limitation='Cosine screen is not proof of specimen provenance independence. Source is newly downloaded; sample provenance retained from authors.'))
 ledger=raw[['Spectrum_ID','Sample_ID','Replica']].merge(meta[['Sample_ID','Type','Subtype','truth','known','partition']],on='Sample_ID',validate='many_to_one',sort=False)
 assert ledger.Spectrum_ID.tolist()==raw.Spectrum_ID.tolist()
 ledger['weight']=1/ledger.groupby('Sample_ID').Spectrum_ID.transform('size')
 rows=[]
 for name,(pred,score) in outputs.items():
  frame=ledger.copy();frame['method']=name;frame['prediction']=pred;frame['score']=score;rows.append(frame)
 scores=pd.concat(rows,ignore_index=True);scores.to_csv(O/'scores.csv',index=False)
 # Only calibration partition is used to choose deployed thresholds.
 thresholds={}
 for name,g in scores[scores.partition.eq('calibration')].groupby('method'):
  t,p,ok=threshold(g.score.to_numpy(),g.weight.to_numpy());assert ok
  thresholds[name]={'threshold':t,'tie_acceptance_probability':p,'calibration_known_acceptance':.9}
 dump('thresholds.json',thresholds)
 results=[];families=[];accepted=[]
 for name,g in scores[scores.partition.eq('evaluation')].groupby('method'):
  k=g.known.to_numpy(bool);v=g.score.to_numpy();w=g.weight.to_numpy();error=g.prediction.ne(g.truth).to_numpy()
  settings=[('calibrated',.9,thresholds[name]['threshold'],thresholds[name]['tie_acceptance_probability'])]
  for target in [.5,.8,.9,.95]:
   t,p,ok=threshold(v[k],w[k],target);assert ok;settings.append(('retrospective_matched',target,t,p))
  for mode,target,t,p in settings:
   acc=accept(v,t,p)
   ka=float(np.sum(w[k]*acc[k])/w[k].sum());ua=float(np.sum(w[~k]*acc[~k])/w[~k].sum());den=np.sum(w[k]*acc[k]);er=float(np.sum(w[k]*acc[k]*error[k])/den) if den else None
   results.append(dict(method=name,evaluation=mode,target=target,known_acceptance=ka,unknown_acceptance=ua,accepted_known_error=er,total_rejection=float(1-np.sum(w*acc)/w.sum())))
   for subtype,z in g.groupby('Subtype'):
    ix=g.index.get_indexer(z.index);ww=w[ix];aa=acc[ix];ee=error[ix]
    families.append(dict(method=name,evaluation=mode,target=target,subtype=subtype,samples=z.Sample_ID.nunique(),known=bool(z.known.iloc[0]),acceptance=float(np.sum(ww*aa)/ww.sum()),wrong_accepted_per_sample=float(np.sum(ww*aa*ee)/ww.sum())))
   if mode=='calibrated':accepted.extend(dict(method=name,Spectrum_ID=int(s),acceptance_weight=float(aa)) for s,aa in zip(g.Spectrum_ID,acc))
 pd.DataFrame(results).to_csv(O/'metrics.csv',index=False);pd.DataFrame(families).to_csv(O/'family_metrics.csv',index=False);pd.DataFrame(accepted).to_csv(O/'acceptance_ledger.csv',index=False)
 assert all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in freeze['hashes'].items())
 dump('verification.json',dict(status='passed',all_160_spectra_retained=len(ledger)==160,all_137_samples_retained=ledger.Sample_ID.nunique()==137,sample_split_disjoint=True,feature_parameters_unchanged=True,refit=False,seconds=time.perf_counter()-start,
  calibration_samples=int(meta.partition.eq('calibration').sum()),evaluation_known_samples=int((meta.known & meta.partition.eq('evaluation')).sum()),evaluation_unknown_samples=int((~meta.known).sum()),
  warning='Small known calibration/evaluation sample counts. Spectra are not independent replicates. No inferential significance claim.'))
 print(pd.DataFrame(results).query('evaluation=="calibrated" or target==0.9').to_string(index=False))

if __name__=='__main__':main()


