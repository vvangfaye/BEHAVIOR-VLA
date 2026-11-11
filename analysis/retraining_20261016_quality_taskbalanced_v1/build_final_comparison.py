"""Publish CSV/HTML only after every fixed20 result passes identity and seed checks."""
import csv
import hashlib
import html
import io
import json
from pathlib import Path

from compare_candidate import validate,summary,paired

ROOT=Path(__file__).resolve().parent
CAMPAIGNS={'mixed1000':'paired_fixed20_step1000_5x4','mixed5000':'paired_fixed20_step5000_5x4','expert1000':'expert_fixed20_step1000_5x4'}
LABELS={'baseline1688977':'原模型 1688977','mixed1000':'混合数据 1,000 步','mixed5000':'混合数据 5,000 步','expert1000':'纯专家 1,000 步'}


def load_complete():
 manifest=json.loads((ROOT/'evaluation_history/manifest.json').read_text())
 pairs=list(map(tuple,manifest['pairs']));models={};proofs=[];missing=[]
 source=manifest['models']['1688977'];path=ROOT/'evaluation_history'/source['file'];raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Pinned baseline changed')
 models['baseline1688977']=validate(json.loads(raw),pairs,manifest['protocol_sha256'],baseline=True)
 proofs.append({'file':str(path.relative_to(ROOT)),'sha256':source['sha256']})
 for name,folder in CAMPAIGNS.items():
  root=ROOT/folder;plan=json.loads((root/'plan.json').read_text());bundle=hashlib.sha256((root/'frozen_sha256.json').read_bytes()).hexdigest();records=[]
  if plan['protocol_sha256']!=manifest['protocol_sha256'] or list(map(tuple,plan['pairs']))!=pairs:raise ValueError('Candidate protocol or panel changed')
  for index,pair in enumerate(pairs):
   path=root/'reports'/f'candidate-{index:02d}.json'
   if not path.exists():missing.append([name,index]);continue
   raw=path.read_bytes();report=json.loads(raw)
   if report.get('state')!='accepted':missing.append([name,index]);continue
   if (report.get('id')!=f'candidate-{index:02d}' or report.get('worker_bundle_sha256')!=bundle
       or report.get('protocol_sha256')!=manifest['protocol_sha256']
       or report.get('checkpoint_manifest_sha256')!=plan['expected_checkpoint_manifest_sha256']):raise ValueError('Candidate identity mismatch')
   if (report['record']['task'],report['record']['instance_id'])!=pair:raise ValueError('Candidate assigned to wrong instance')
   records.append(report['record']);proofs.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'recovery':report.get('recovery_provenance')})
  if len(records)==20:models[name]=validate({'protocol_sha256':manifest['protocol_sha256'],'records':records},pairs,manifest['protocol_sha256'])
 return pairs,models,proofs,missing


def main():
 pairs,models,proofs,missing=load_complete()
 if missing:
  print(json.dumps({'published':False,'reason':'awaiting_all_20_in_each_campaign','missing_count':len(missing)}));return
 if len(pairs)!=20 or set(models)!=set(LABELS):raise ValueError('Full fixed20 comparison unavailable')
 training_path=ROOT/'training_comparability.json'
 training_raw=training_path.read_bytes();training=json.loads(training_raw)
 for source in training['source_proofs']:
  if hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()!=source['sha256']:
   raise ValueError('Frozen training comparability source changed')
 sampling_path=ROOT/'matched1000_sampling_audit.json'
 sampling_raw=sampling_path.read_bytes();sampling=json.loads(sampling_raw)
 for path,expected in sampling['source_sha256'].items():
  if hashlib.sha256((ROOT/'formal_stage/bundle'/path).read_bytes()).hexdigest()!=expected:
   raise ValueError('Frozen sampling source changed')
 if hashlib.sha256((ROOT/'training_metrics.snapshot.jsonl').read_bytes()).hexdigest()!=sampling['metrics_snapshot_sha256']:
  raise ValueError('Sampling runtime evidence changed')
 incident_path=ROOT/'recovery_incident_summary.json'
 incident_raw=incident_path.read_bytes();incidents=json.loads(incident_raw)
 for source in incidents['source_proofs']:
  if hashlib.sha256((ROOT/source['file']).read_bytes()).hexdigest()!=source['sha256']:
   raise ValueError('Recovery incident evidence changed')
 stats={name:summary(rows,pairs) for name,rows in models.items()}
 comparisons={name+'_vs_baseline':paired(models['baseline1688977'],models[name],pairs) for name in CAMPAIGNS}
 comparisons['expert1000_vs_mixed1000']=paired(models['mixed1000'],models['expert1000'],pairs)
 comparisons['expert1000_vs_mixed5000']=paired(models['mixed5000'],models['expert1000'],pairs)
 fields=['task','instance']
 for name in LABELS:fields.extend([name+'_success',name+'_q',name+'_steps'])
 fields+=['expert1000_minus_mixed1000_q','expert1000_minus_mixed5000_q']
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();body=[]
 for task,instance in pairs:
  row={'task':task,'instance':instance};cells=[html.escape(task),str(instance)]
  for name in LABELS:
   r=models[name][(task,instance)];row.update({name+'_success':r['model_success'],name+'_q':r['q_score'],name+'_steps':r['steps']})
   cells.append(('成功' if r['model_success'] else '未成功')+' / '+f"{r['q_score']:.4f}")
  for other in ('mixed1000','mixed5000'):row['expert1000_minus_'+other+'_q']=models['expert1000'][(task,instance)]['q_score']-models[other][(task,instance)]['q_score']
  writer.writerow(row);body.append('<tr>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
 csv_data=stream.getvalue().encode('utf-8-sig')
 summaries=''.join('<tr><td>'+LABELS[n]+'</td><td>'+str(stats[n]['successes'])+'/20</td><td>'+f"{stats[n]['mean_q']:.4f}"+'</td></tr>' for n in LABELS)
 contrasts=''.join('<tr><td>'+html.escape(n)+'</td><td>'+f"{v['mean_q_delta']:+.4f}"+'</td><td>'+str(v['paired_q_bootstrap95'])+'</td><td>'+str(len(v['success_gains']))+'/'+str(len(v['success_losses']))+'</td></tr>' for n,v in comparisons.items())
 document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定20组完整对比</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eef3f7}h1{font-size:28px}</style><h1>固定20组完整对比</h1><p>所有模型使用相同实例和种子。两组新实验均已验收20/20，基础设施失败没有计为模型失败，补跑只覆盖未完成样本。</p><table><tr><th>模型</th><th>成功数</th><th>平均任务进度</th></tr>'''+summaries+'''</table><p>纯专家1,000步与混合1,000步的对比控制了更新次数；纯专家1,000步与混合5,000步同时存在数据和训练步数差异，不能单独归因于数据配比。该固定面板含训练见过的实例，不代表独立泛化结果。</p><table><tr><th>配对比较</th><th>平均进度差</th><th>固定面板bootstrap 95%区间</th><th>新增成功/失去成功</th></tr>'''+contrasts+'''</table><p>每格为“成功状态 / 任务进度分数”。逐样本步数、精确分数及差值见CSV。</p><table><tr><th>任务</th><th>实例</th>'''+''.join('<th>'+LABELS[n]+'</th>' for n in LABELS)+'''</tr>'''+''.join(body)+'''</table></html>'''
 (ROOT/'final_comparison.csv').write_bytes(csv_data)
 document=document.replace('</table><p>纯专家1,000步', '</table><p>已核验冻结训练配置：两组初始化、优化器、学习率日程、归一化、全局批大小和随机种子一致。</p><p>纯专家1,000步', 1)
 document=document.replace('<p>每格为', '<p>前1,000步共192,000个训练窗口：混合组182,250个专家窗口、9,750个完整轨迹窗口（5.078125%）；纯专家组192,000个专家窗口。两组逐位置任务一致，原有专家窗口不变。采样计数与混合组训练日志一致；未据此声称数据增强随机数相同。</p><p>每格为', 1)
 document=document.replace('</html>', '<h2>失败与恢复</h2><p>最初两个混合批次在启动脚本及样本产物落盘前失败，旧平台日志已不可用，底层原因未能确定。恢复时改为预先写入AFS并校验启动文件。后续卡住的样本有模拟器更新调用中OmniGraph停滞的栈证据；停止旧进程后用新worker仅补跑未完成样本，保持权重、种子、源码和原生帧预算不变。这是恢复措施，不表示已修复模拟器内部问题。基础设施失败没有按模型失败计分。</p></html>')
 (ROOT/'final_comparison.html').write_text(document)
 proof={'complete':True,'protocol_and_seed_checks_passed':True,'summaries':stats,'comparisons':comparisons,'input_proofs':proofs,'training_comparability':{'file':training_path.name,'sha256':hashlib.sha256(training_raw).hexdigest(),'details':training},'generalization_proven':False,'output_sha256':{'csv':hashlib.sha256(csv_data).hexdigest(),'html':hashlib.sha256(document.encode()).hexdigest()}}
 proof['matched1000_sampling']={'file':sampling_path.name,'sha256':hashlib.sha256(sampling_raw).hexdigest(),'details':sampling}
 proof['recovery_incidents']={'file':incident_path.name,'sha256':hashlib.sha256(incident_raw).hexdigest(),'details':incidents}
 (ROOT/'final_comparison_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'published':True,'files':['final_comparison.csv','final_comparison.html','final_comparison_proof.json']}))


if __name__=='__main__':main()
