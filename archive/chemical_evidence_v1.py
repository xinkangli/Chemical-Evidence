"""Measured band checks with explicit observability; no chemical identity decisions."""
import hashlib,json
import numpy as np
SOURCE='https://doi.org/10.1002/jemt.23841'
# Reported peak positions, Table 1. Bands are not unique identifiers or universal mandatory peaks.
SIGNATURES={'polyethylene':[2915,2845,1462,717],
 'polypropylene':[2915,1455,1377,997,982],
 'polystyrene':[3024,2847,1492,1451,694],
 'polyethylene_terephthalate':[1713,1241,1094,720]}
PARAMS={'search_half_width_cm1':12,'baseline_inner_cm1':24,'baseline_outer_cm1':42,
 'max_gap_cm1':12,'min_search_points':3,'min_flank_points':2,'min_contrast_fraction':.02,'min_noise_proxy_ratio':3.,
 'status':'Frozen engineering feature checks, not validated limits of chemical detection; no material exclusion from a nondetection.'}

def check_spectrum(axis,y,mode,signal_type,observation_id):
 axis=np.asarray(axis,float);y=np.asarray(y,float)
 if axis.ndim!=1 or y.shape!=axis.shape or len(axis)<3 or not np.isfinite(axis).all() or not np.isfinite(y).all():raise ValueError('invalid spectrum')
 if np.all(np.diff(axis)<0):axis=axis[::-1];y=y[::-1]
 if not np.all(np.diff(axis)>0):raise ValueError('nonmonotonic or duplicate axis')
 original=y.copy();recognized=signal_type in ['absorbance','reported_absorbance','transmittance_fraction','transmittance_percent']
 if signal_type.startswith('transmittance'):
  t=y/(100 if signal_type=='transmittance_percent' else 1)
  if np.any(t<=0) or np.any(t>1):raise ValueError('invalid transmittance; no clipping or inferred scaling')
  y=-np.log10(t)
 scale=float(np.quantile(y,.99)-np.quantile(y,.01));noise=float(np.median(np.abs(np.diff(y,n=2)-np.median(np.diff(y,n=2))))/(.67448975*np.sqrt(6)))
 rows=[]
 for family,centers in SIGNATURES.items():
  for center in centers:
   eid=observation_id+':'+family+':'+str(center)
   r={'evidence_id':eid,'observation_id':observation_id,'family_hypothesis':family,'reference_center_cm1':center,
    'source_url':SOURCE,'source_locator':'Table 1','reference_mode':'ATR','query_mode':mode,'mode_relation':'same_declared_mode' if mode=='ATR' else 'cross_mode_positions_only' if mode=='Transmission' else 'unknown',
    'signal_type':signal_type,'sampling_step_median_cm1':float(np.median(np.diff(axis))),'instrument_resolution_cm1':None,
    'status':'unassessable','reason':None,'observed_peak_cm1':None,'contrast_fraction':None,'noise_proxy_ratio':None,'peak_observation_id':None,
    'absence_claim_permitted':False,'unique_material_identity_claim_permitted':False}
   lo=center-PARAMS['baseline_outer_cm1'];hi=center+PARAMS['baseline_outer_cm1']
   if center<axis[0] or center>axis[-1]:r['reason']='reference_center_outside_observed_range'
   elif lo<axis[0] or hi>axis[-1]:r['reason']='insufficient_baseline_context_at_range_edge'
   elif not recognized:r['reason']='signal_convention_unknown'
   elif mode not in ['ATR','Transmission']:r['reason']='query_measurement_mode_unknown'
   elif scale<=1e-12:r['reason']='flat_or_degenerate_signal'
   else:
    core=np.flatnonzero(np.abs(axis-center)<=12);left=np.flatnonzero((axis>=lo)&(axis<=center-24));right=np.flatnonzero((axis>=center+24)&(axis<=hi));local=axis[(axis>=lo)&(axis<=hi)]
    if len(core)<3 or len(left)<2 or len(right)<2:r['reason']='too_few_measured_points'
    elif np.max(np.diff(local))>12:r['reason']='internal_sampling_gap'
    else:
     xl,xr=np.median(axis[left]),np.median(axis[right]);yl,yr=np.median(y[left]),np.median(y[right]);baseline=yl+(axis[core]-xl)*(yr-yl)/(xr-xl)
     residual=y[core]-baseline;j=int(np.argmax(residual));idx=int(core[j]);height=float(residual[j]);fraction=height/scale;ratio=height/max(noise,scale*1e-8)
     # A boundary maximum is not a resolved local feature within this search interval.
     localmax=0<j<len(core)-1 and residual[j]>residual[j-1] and residual[j]>=residual[j+1]
     detected=localmax and fraction>=.02 and ratio>=3
     r.update(status='local_feature_detected' if detected else 'no_resolved_feature',reason='passes_fixed_feature_checks' if detected else 'fixed_feature_checks_not_met_not_proof_of_absence',
      observed_peak_cm1=float(axis[idx]) if detected else None,contrast_fraction=float(fraction),noise_proxy_ratio=float(ratio),
      peak_observation_id=observation_id+':measured_index:'+str(idx) if detected else None)
   rows.append(r)
 return {'observation_id':observation_id,'query_mode':mode,'signal_type':signal_type,'axis_min':float(axis.min()),'axis_max':float(axis.max()),'points':len(axis),
  'measured_signal_sha256':hashlib.sha256(np.column_stack([axis,original]).tobytes()).hexdigest(),
  'noise_estimate_status':'second-difference roughness proxy, not measured instrumental noise or validated SNR','bands':rows}

def summarize_candidate(records,family):
 bands=[b for r in records for b in r['bands'] if b['family_hypothesis']==family]
 found=[b for b in bands if b['status']=='local_feature_detected'];unassessed=[b for b in bands if b['status']=='unassessable']
 return {'family_hypothesis':family,'expected_checks':len(bands),'detected_checks':len(found),'unique_peak_observations':len(set(b['peak_observation_id'] for b in found)),
  'unassessable_checks':len(unassessed),'not_resolved_checks':sum(b['status']=='no_resolved_feature' for b in bands),
  'supporting_feature_ids':[b['evidence_id'] for b in found],'unassessable_ids':[b['evidence_id'] for b in unassessed],
  'interpretation':'Observed local features near literature positions; shared bands and correlated repeats are not independent proof of identity. Nondetection cannot exclude a material.',
  'material_identity_verified':False,'purity_or_mixture_verified':False}

def self_tests():
 x=np.arange(600.,3301.,1);y=np.exp(-.5*((x-1713)/4)**2)
 def band(packet,family='polyethylene_terephthalate',center=1713):return next(b for b in packet['bands'] if b['family_hypothesis']==family and b['reference_center_cm1']==center)
 p=check_spectrum(x,y,'ATR','absorbance','test');assert band(p)['status']=='local_feature_detected'
 mask=(x>=702)&(x<=1800);crop=check_spectrum(x[mask],y[mask],'ATR','absorbance','crop');assert band(crop,'polystyrene',694)['reason']=='reference_center_outside_observed_range'
 assert band(check_spectrum(x,y,'ATR','unknown','u'))['reason']=='signal_convention_unknown'
 assert band(check_spectrum(x,y,'unknown','absorbance','m'))['reason']=='query_measurement_mode_unknown'
 assert band(check_spectrum(x,np.zeros_like(x),'ATR','absorbance','f'))['status']=='unassessable'
 rev=check_spectrum(x[::-1],y[::-1],'ATR','absorbance','rev');assert band(rev)['observed_peak_cm1']==1713
 trans=check_spectrum(x,100*10**(-y),'Transmission','transmittance_percent','t');assert band(trans)['observed_peak_cm1']==1713
 assert band(trans)['mode_relation']=='cross_mode_positions_only'
 gap=(x<1705)|(x>1720);assert band(check_spectrum(x[gap],y[gap],'ATR','absorbance','gap'))['reason']=='internal_sampling_gap'
 for bad in [np.full_like(y,np.nan),np.full_like(y,np.inf)]:
  try:check_spectrum(x,bad,'ATR','absorbance','bad')
  except ValueError:pass
  else:raise AssertionError('nonfinite allowed')
 q=check_spectrum(x,np.exp(-.5*((x-718)/4)**2),'ATR','absorbance','shared');a=band(q,'polyethylene',717);b=band(q,'polyethylene_terephthalate',720);assert a['peak_observation_id']==b['peak_observation_id']
 assert not any(r['absence_claim_permitted'] or r['unique_material_identity_claim_permitted'] for r in p['bands'])
 return {'status':'passed','checks':11,'scope':'engineering behavior only; not chemical accuracy validation'}

if __name__=='__main__':print(json.dumps(self_tests()))


