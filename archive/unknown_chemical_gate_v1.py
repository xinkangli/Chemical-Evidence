"""Bounded retrospective stop/go diagnostic; not independent confirmation."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/unknown_chemical_gate_v1'

def main():
 OUT.mkdir(exist_ok=False)
 inputs=[ROOT/'reports/calibrated_evidence_fusion_v1/case_results.csv',ROOT/'reports/chemical_evidence_v1/band_evidence.parquet',Path(__file__)]
 protocol=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='retrospective development stop/go',
  primary_known_acceptance=.9,secondary=[.5,.8,.95],unknown='All existing PLA cases; one unknown family only',
  candidate='fixed_fusion unchanged for chemical methods',scores='distinct detected peaks / candidate reference count; shared peaks weighted by inverse number of matching families; equal mean percentile-rank fusion with fixed-fusion confidence',
  ties='Fractional acceptance of boundary ties, expected randomized rule; no label-dependent within-tie selection',
  thresholds='Retrospective known-score quantile; NOT deployable independently calibrated thresholds',
  stop_go='Continue only if at primary coverage a chemical method has lower unknown acceptance than EVERY existing baseline, without higher accepted-known error than fixed_fusion. Even a pass requires independent confirmation.',
  limits=['All cases previously viewed','One unknown product group; no spectrum-level significance or independent confidence interval','Missing Cowger signal convention prevents chemical evaluation; all 128 cases retained in scope audit','No new LLM or model training; cannot establish an agent contribution'],
  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 d=pd.read_csv(inputs[0]); all_cases=d.drop_duplicates('case_id')
 (OUT/'scope.json').write_text(json.dumps({'total_cases':len(all_cases),'chemical_evaluable':3789,'units_unresolved':128,'excluded_by_performance':0},indent=2))
 d=d[d.panel.eq('MicroPlastiX')].copy()
 base=d[d.method.eq('fixed_fusion')].set_index('case_id').sort_index()
 b=pd.read_parquet(inputs[1]); b=b[b.case_id.isin(base.index)]
 scores=[]
 for cid,g in b.groupby('case_id'):
  detected=g[g.status.eq('local_feature_detected')]
  famcount=detected.groupby('peak_observation_id').family_hypothesis.nunique()
  fam=base.loc[cid,'prediction']; checks=g[g.family_hypothesis.eq(fam)]
  peaks=detected[detected.family_hypothesis.eq(fam)].peak_observation_id.dropna().unique()
  scores.append({'case_id':cid,'chemical_support':len(peaks)/len(checks),
   'chemical_specificity':sum(1/famcount[p] for p in peaks)/len(checks)})
 s=pd.DataFrame(scores).set_index('case_id').reindex(base.index)
 assert len(s)==3789 and not s.isna().any().any()
 extra=[]
 for name in s:
  x=base.copy(); x['method']=name; x['score']=s[name];extra.append(x.reset_index())
  x=x.copy();x['method']=name+'_rank_fusion';x['score']=(s[name].rank(pct=True)+base.score.rank(pct=True))/2;extra.append(x.reset_index())
 d=pd.concat([d]+extra,ignore_index=True)
 assert d.groupby('method').size().eq(3789).all()
 assert d.groupby('method').known.sum().eq(3342).all()
 d.to_parquet(OUT/'scores.parquet',index=False)
 rows=[];families=[]; weights=[]
 for method,g in d.groupby('method'):
  g=g.copy(); k=g.known.to_numpy(bool); v=g.score.to_numpy(); err=g.prediction.ne(g.truth).to_numpy()
  for target in [.5,.8,.9,.95]:
   vals=np.sort(v[k])[::-1]; n=target*k.sum(); threshold=vals[int(np.ceil(n))-1]
   above=v>threshold;equal=v==threshold
   fraction=(n-np.sum(above&k))/np.sum(equal&k)
   w=above.astype(float)+equal*fraction
   assert abs(w[k].sum()/k.sum()-target)<1e-10 and 0<=fraction<=1
   rows.append(dict(method=method,known_acceptance=target,unknown_acceptance=w[~k].mean(),
    accepted_known_error=(w[k]*err[k]).sum()/w[k].sum(),known_correct_accepted=(w[k]*(~err[k])).mean(),
    total_rejection=1-w.mean(),threshold=threshold,boundary_probability=fraction))
   for fam in g.truth.unique():
    mask=g.truth.eq(fam).to_numpy();families.append(dict(method=method,known_acceptance=target,family=fam,n=int(mask.sum()),acceptance=w[mask].mean(),wrong_accepted_per_case=(w[mask]*err[mask]).mean()))
   if target==.9:
    weights.extend(dict(case_id=c,method=method,acceptance_weight=float(a)) for c,a in zip(g.case_id,w))
 r=pd.DataFrame(rows);r.to_csv(OUT/'metrics.csv',index=False)
 pd.DataFrame(families).to_csv(OUT/'family_metrics.csv',index=False)
 pd.DataFrame(weights).to_csv(OUT/'primary_case_weights.csv',index=False)
 primary=r[r.known_acceptance.eq(.9)].set_index('method')
 baseline=primary[~primary.index.str.startswith('chemical_')]
 candidate=primary[primary.index.str.startswith('chemical_')]
 passing=candidate[(candidate.unknown_acceptance<baseline.unknown_acceptance.min()) & (candidate.accepted_known_error<=primary.loc['fixed_fusion','accepted_known_error'])]
 result=dict(status='continue_to_independent_confirmation' if len(passing) else 'stop_current_chemical_gate_route',passing_methods=passing.index.tolist(),best_baseline_unknown_acceptance=float(baseline.unknown_acceptance.min()),new_training_runs=0,llm_calls=0,unknown_cases=447,known_cases=3342,unknown_independent_families=1)
 (OUT/'decision.json').write_text(json.dumps(result,indent=2))
 print(primary[['unknown_acceptance','accepted_known_error']].to_string());print(json.dumps(result))

if __name__=='__main__':main()


