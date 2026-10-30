"""Recompute four-source tables from archived per-observation scores; no refitting."""
from pathlib import Path
import argparse
import pandas as pd
from chemical_evidence.evaluation import metrics,matched_weights
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[];families=[]
 for path in sorted((ROOT/'data/scores').glob('*.parquet')):
  frame=pd.read_parquet(path)
  for method,g in frame.groupby('method'):
   for target in ([.5,.8,.9,.95] if path.stem=='MPX' else [.9]):
    result=metrics(g.score,g.known,g.prediction.eq(g.truth),target)
    rows.append(dict(panel=path.stem,method=method,**result))
    if target==.9:
     w,_,_=matched_weights(g.score,g.known,target);g=g.assign(acceptance=w)
     for material,sub in g.groupby('reported_label'):
      families.append(dict(panel=path.stem,method=method,material=material,n=len(sub),acceptance=sub.acceptance.mean()))
 result=pd.DataFrame(rows);result.to_csv(out/'metrics.csv',index=False)
 pd.DataFrame(families).to_csv(out/'class_acceptance.csv',index=False)
 expected=pd.read_csv(ROOT/'data/expected_primary_metrics.csv')
 got=result[result.known_acceptance.sub(.9).abs()<1e-10]
 merged=expected.merge(got,on=['panel','method'],suffixes=('_expected','_actual'),validate='one_to_one')
 assert len(merged)==len(expected)==36
 for metric in ['unknown_acceptance','accepted_known_error']:
  assert (merged[metric+'_expected']-merged[metric+'_actual']).abs().max()<1e-10,metric
 print('Verified all 36 primary method/source rows against archived metrics.')
 print(got[got.method.isin(['fixed_fusion','chemical_support'])].to_string(index=False))
if __name__=='__main__':main()
