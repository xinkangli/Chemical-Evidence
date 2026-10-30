"""Build traceable chemical-band evidence from all existing development cases."""
from pathlib import Path
import datetime,json,hashlib,zipfile,io,time,gzip
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
import chemical_evidence_v1 as ce
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/chemical_evidence_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 start=time.perf_counter();OUT.mkdir(exist_ok=False);(OUT/'packets').mkdir();(OUT/'pilot_cards').mkdir()
 paths=[Path(__file__),ROOT/'code/chemical_evidence_v1.py',ROOT/'reports/paired_calibration_v1/aggregate_records.parquet',ROOT/'reports/public_mpx_v1/records.parquet',ROOT/'reports/calibrated_evidence_fusion_v1/case_results.csv',ROOT/'data/raw/mpx_1820/spectra.zip',ROOT/'data/curated/v20260928_03/openspecy/spectra.parquet']
 dump(OUT/'protocol_freeze.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'Development tool implementation on previously viewed sources; not identification or confirmation benchmark.',
  'cohort':'All existing 64 Cowger test material IDs in both modes, ALL constituent records; all 3789 MicroPlastiX records. No success-based exclusions.',
  'reference':{'url':ce.SOURCE,'locator':'Table 1','role':'Reported polymer peak positions, not universal mandatory or unique diagnostic bands'},'signatures':ce.SIGNATURES,'parameters':ce.PARAMS,
  'units':'Use only explicit per-record intensity_type or raw CSV headers and metadata. Unspecified units remain unknown, even when a general source script labels a variable Absorbance.',
  'scope':'Measure local signal features. Never infer absence, purity, mixture composition, complete structure or validated identity. Reference measured in ATR; Transmission is cross-mode positional comparison only.',
  'pilot_selection':'At most one MicroPlastiX spectrum per Polymer_ID, prefer initial/fixed-fusion disagreement without looking at truth, then SHA256(case_id). Cowger per-mode smallest case hash. Pilot materials are NOT independent confirmation.',
  'comparison':'Full available axis versus same raw scan cropped to 702..1800; paired feature observability only, not classification gain. Cowger repeats kept separate and never counted as independent materials.',
  'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
 dump(OUT/'unit_tests.json',ce.self_tests())
 a=pd.read_parquet(paths[2]);m=pd.read_parquet(paths[3]);old=pd.read_csv(paths[4],usecols=['case_id','method','prediction','score']);old=old[old.method.isin(['initial_retrieval','fixed_fusion'])]
 preds={k:v.set_index('method').prediction.to_dict() for k,v in old.groupby('case_id')}
 wanted=set(a[a.partition.eq('test')].record_ids.explode());raw={};p=pq.ParquetFile(paths[-1])
 # Read cheap IDs first, decode signals only for row groups containing wanted records.
 for i in range(p.num_row_groups):
  ids=p.read_row_group(i,columns=['record_id']);hits=set(ids['record_id'].to_pylist())&wanted
  if not hits:continue
  cols=['record_id','axis','intensity','intensity_type','measurement_mode','axis_unit','spectral_sha256'];t=p.read_row_group(i,columns=cols);t=t.filter(pc.is_in(t['record_id'],value_set=pa.array(sorted(hits))))
  raw.update({r['record_id']:r for r in t.to_pylist()})
 assert set(raw)==wanted
 specs=[];provenance=[];case_meta=[]
 for r in a[a.partition.eq('test')].itertuples():
  cid='C-'+r.mode+'-'+str(r.ID);obs=[]
  for rid in sorted(r.record_ids):
   q=raw[rid];oid='obs-'+hashlib.sha256(rid.encode()).hexdigest()[:20];signal={'absorbance':'absorbance','A':'absorbance','transmittance':'unknown','unspecified':'unknown'}.get(q['intensity_type'],'unknown')
   mode={'atr':'ATR','transmission':'Transmission'}.get(q['measurement_mode'],'unknown');assert mode==r.mode and q['axis_unit']=='cm^-1'
   obs.append(oid);specs.append((cid,oid,np.asarray(q['axis']),np.asarray(q['intensity']),mode,signal))
   provenance.append({'observation_id':oid,'case_id':cid,'source_record_id':rid,'source':'Cowger via OpenSpecy','signal_convention_evidence':q['intensity_type'],'unit_status':'unresolved' if signal=='unknown' else 'explicit_metadata'})
  case_meta.append({'case_id':cid,'panel':'Cowger_'+r.mode,'cluster':'Cowger-'+str(r.ID),'observations':obs})
 z=zipfile.ZipFile(paths[5]);assert z.testzip() is None
 for i,r in m.iterrows():
  cid='MPX-'+str(i);oid='obs-'+hashlib.sha256(r['file'].encode()).hexdigest()[:20];tab=pd.read_csv(io.BytesIO(z.read(r['file'])));arr=tab.to_numpy(float)
  assert r['x_unit']=='cm-1' and r['y_unit']=='A' and tab.columns.tolist()==['cm-1','A'];order=np.argsort(arr[:,0]);arr=arr[order]
  specs.append((cid,oid,arr[:,0],arr[:,1],'ATR','absorbance'));provenance.append({'observation_id':oid,'case_id':cid,'source_record_id':r['file'],'source':'MicroPlastiX','signal_convention_evidence':'Metadata y_unit=A AND CSV header A','unit_status':'explicit_metadata_and_file'})
  case_meta.append({'case_id':cid,'panel':'MicroPlastiX','cluster':'MPX-'+str(r.Polymer_ID),'observations':[oid]})
 assert len(case_meta)==3917
 meta={r['case_id']:r for r in case_meta};bycase={cid:[] for cid in meta};bandrows=[];coverage=[];plot_data={};source_rows=[]
 # Hash based pilot selection, without chemical truth or new feature outcomes.
 for panel in ['Cowger_ATR','Cowger_Transmission','MicroPlastiX']:
  eligible=[r for r in case_meta if r['panel']==panel];groups={}
  for r in eligible:groups.setdefault(r['cluster'],[]).append(r)
  ordered=[]
  for cluster,rs in sorted(groups.items()):
   rs.sort(key=lambda r:(preds[r['case_id']]['initial_retrieval']==preds[r['case_id']]['fixed_fusion'],hashlib.sha256(r['case_id'].encode()).hexdigest()))
   ordered.append(rs[0])
  if panel!='MicroPlastiX':ordered=sorted(ordered,key=lambda r:hashlib.sha256(r['case_id'].encode()).hexdigest())[:1]
  source_rows.extend(ordered)
 pilot_ids={r['case_id'] for r in source_rows}
 for cid,oid,x,y,mode,signal in specs:
  packet=ce.check_spectrum(x,y,mode,signal,oid);bycase[cid].append(packet)
  mask=(x>=702)&(x<=1800);cropped=ce.check_spectrum(x[mask],y[mask],mode,signal,oid+':crop')
  for kind,pk in [('full',packet),('cropped',cropped)]:
   for b in pk['bands']:
    coverage.append({'case_id':cid,'observation_id':oid,'panel':meta[cid]['panel'],'cluster':meta[cid]['cluster'],'window':kind,'family':b['family_hypothesis'],'center':b['reference_center_cm1'],'status':b['status'],'reason':b['reason']})
  bandrows.extend({'case_id':cid,**b} for b in packet['bands'])
  if cid in pilot_ids and cid not in plot_data:plot_data[cid]=(x,y,oid,signal)
 pd.DataFrame(provenance).to_csv(OUT/'provenance.csv',index=False);pd.DataFrame(bandrows).to_parquet(OUT/'band_evidence.parquet',index=False)
 cv=pd.DataFrame(coverage);cv.to_parquet(OUT/'window_comparison.parquet',index=False);cv.groupby(['panel','window','status','reason']).size().rename('band_checks').reset_index().to_csv(OUT/'observability_summary.csv',index=False)
 scalar=[]
 for cid,rs in bycase.items():
  cands=[ce.summarize_candidate(rs,f) for f in ce.SIGNATURES];packet={'case_id':cid,'mode':rs[0]['query_mode'],'initial_hypothesis':preds[cid]['initial_retrieval'],'alternative_hypothesis':preds[cid]['fixed_fusion'],
   'scope':'Chemical review packet, not an identity verdict. Separate observations from same material are correlated. No test truth supplied.',
   'observation_checks':rs,'candidate_checks':cands,'required_report_items':['initial and competing hypothesis','supporting measured features with evidence IDs','unassessable regions or metadata','conflicts and nonspecificity','unresolved conclusion and needed verification']}
  dump(OUT/'packets'/(cid+'.json'),packet)
  scalar.append({'case_id':cid,'panel':meta[cid]['panel'],'cluster':meta[cid]['cluster'],'raw_observations':len(rs),'unknown_signal_observations':sum(r['signal_type']=='unknown' for r in rs),'full_min':min(r['axis_min'] for r in rs),'full_max':max(r['axis_max'] for r in rs)})
 pd.DataFrame(scalar).to_csv(OUT/'case_manifest.csv',index=False)
 make_pilot(source_rows,plot_data,bycase,preds)
 dump(OUT/'verification.json',{'status':'passed','cases':3917,'raw_observations':len(specs),'cowger_raw_observations':len(raw),'unknown_units_preserved':sum(s[-1]=='unknown' for s in specs),'band_checks':len(bandrows),'pilot_cases':len(source_rows),'truth_in_packets':False,'independent_confirmation':False,'seconds':time.perf_counter()-start})
 print(json.dumps(json.loads((OUT/'verification.json').read_text()),ensure_ascii=False));print('CHEMICAL_EVIDENCE_COMPLETE',flush=True)

def make_pilot(rows,plot_data,bycase,preds):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cards=[]
 for j,r in enumerate(rows,1):
  cid=r['case_id'];token='DEV-'+str(j).zfill(2);x,y,oid,signal=plot_data[cid];fig,ax=plt.subplots(figsize=(9,3.3),layout='constrained');ax.plot(x,y,lw=.8,color='#23455e');ax.axvspan(702,1800,color='#999999',alpha=.12,label='Previous classifier window')
  ax.set(xlim=(min(4000,max(x)),min(x)),xlabel='Wavenumber (cm$^{-1}$)',ylabel='Absorbance' if signal=='absorbance' else 'Supplied intensity (unit unresolved)',title=token+' | '+r['panel']);ax.legend(fontsize=8)
  fig.savefig(OUT/'pilot_cards'/(token+'.png'),dpi=160);plt.close(fig)
  lines=[f'# {token}：开发复核材料','',f'初始候选：{preds[cid]["initial_retrieval"]}；参考融合候选：{preds[cid]["fixed_fusion"]}。',
   '',f'原始测量记录数：{len(bycase[cid])}。图中展示第一条记录；完整检查包含全部记录，重复测量不视为独立样品。',
   '',f'![光谱]({token}.png)','','|候选|已检出的局部特征检查|无法评估|未解析出特征|','|---|---:|---:|---:|']
  for f in ce.SIGNATURES:
   s=ce.summarize_candidate(bycase[cid],f);lines.append(f'|{f}|{s["detected_checks"]}|{s["unassessable_checks"]}|{s["not_resolved_checks"]}|')
  lines+=['','以上数字是检查条目，既不是独立证据票数，也不是鉴定概率。同一峰可对应多个材料、邻近参考峰和重复记录。','',
   '请核验：支持候选的谱段是否真实存在；未观测和未解析出峰是否被正确区分；参考测量条件是否适用；现有信息是否足以改变材料身份结论。',
   '',f'完整机器证据：[JSON](../packets/{cid}.json)。参考峰位来自 https://doi.org/10.1002/jemt.23841 Table 1。','',
   '**此包用于开发复核流程，不是方法盲评结果，也不是独立确认。尚无独立专家评分，不得填入假定专家分数。**']
  (OUT/'pilot_cards'/(token+'.md')).write_text('\n'.join(lines),encoding='utf-8');cards.append({'token':token,'case_id':cid,'panel':r['panel'],'cluster':r['cluster']})
 pd.DataFrame(cards).to_csv(OUT/'pilot_case_key.csv',index=False)
 with (OUT/'pilot_review_form.jsonl').open('w',encoding='utf-8') as f:
  for reviewer in ['R1','R2']:
   for c in cards:f.write(json.dumps({'reviewer_id':reviewer,'case_token':c['token'],'start_time':None,'end_time':None,'active_review_seconds':None,'needed_assertions_complete':None,'unsupported_assertions':None,'incorrect_assertions':None,'unassessable_assertions':None,'accept_or_revise':None,'correction_notes':None},ensure_ascii=False)+'\n')
 text='''# 化学核查工具开发报告

本轮新增的是实际谱段检查，不更新分类模型，也不报告尚未验证的识别提升。全部原有3917条任务保留；Cowger按原始记录逐条检查，MicroPlastiX从原始CSV恢复完整波数范围。既有裁剪窗口只作成对可观测性对照。

## 工具做了什么

以文献报告峰位为待检查位置，保存实测局部特征、来源、测量模式、强度约定、采样与边界状态。检测阈值是开发用工程规则，并非经验证的化学检出限。高频噪声估计只是粗糙度代理，不能写成实测信噪比。

Cowger记录的强度类型字段为unspecified；尽管作者脚本使用Absorbance变量名，这不能单独证明每条转存谱的单位。因此保留原始数据和测量方式，将相关化学判断标记为无法评估，未按曲线形状猜单位。MicroPlastiX的元数据与CSV列名均标记A，可进行吸收峰的局部检查。

缺峰、峰重叠、范围缺失与强度类型未知分别记录。工具不允许从未检出局部特征直接宣布材料不存在，不把一个共享峰或多个重复扫描当成独立身份验证，也不推断纯度或混合物组成。

## 产出

- packets/：全部3917个任务的结构化证据，不含测试真值。
- band_evidence.parquet：逐原始观测、逐候选的局部检查。
- window_comparison.parquet：全范围与裁剪范围的成对检查。
- observability_summary.csv：所有状态与失败原因，未删除困难样品。
- pilot_cards/及pilot_review_form.jsonl：两位复核者可用于试用流程的材料与空白记录表。

## 专家评价边界

这些材料是开发流程试用包，尚未匿名随机化为正式方法对比。正式评价应冻结规则、智能体提示词和病例后，分配不同报告版本，隐藏方法名称，并记录核验和修正时间、断言错误与报告完整性。专家偏好不是化学真值；重复使用旧病例也不会产生新测试集。

尚未执行专家评审，尚未证明智能体优于固定规则。下一步先解决Cowger强度语义的来源验证，在可核查的原始数据上形成固定规则与智能体的同信息报告，再冻结正式评审。不要为获得漂亮结果继续改分类器或剔除失败数据。
'''
 (OUT/'REPORT_ZH.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()


