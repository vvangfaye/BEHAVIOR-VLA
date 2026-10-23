import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
p=json.loads((HERE/'storage_plan.json').read_text());c=aoss_client();records=[]
for u in p['units']:
 try:r=c.get_object(Bucket='260010165',Key=u['prefix']+'/RELEASED.json')
 except c.exceptions.NoSuchKey:continue
 try:v=json.loads(r['Body'].read())
 finally:r['Body'].close()
 assert v['archive_id']==p['archive_id'] and v['source']==u['source'] and v['released_files']==len(u['files'])
 records.append({**v,'id':u['id'],'prefix':u['prefix']})
total=sum(r['unlinked_allocated_bytes'] for r in records);state='全部完成' if len(records)==len(p['units']) else '进行中'
lines=['# AFS 清理结果 / 进度（2026-10-23）','',f'状态：{state}。已清理 {len(records)}/{len(p["units"])} 个单元，约 **{total/2**30:.2f} GiB**（已解除文件分配块统计）。','','所有已清理内容均已完整回读校验 SHA256，且源目录保留 `AFS_ARCHIVED.json` 恢复指针。','','| 已清理源目录 | GiB |','|---|---:|']
for r in records:lines.append(f'| `{r["source"]}` | {r["unlinked_allocated_bytes"]/2**30:.2f} |')
lines+=['','运行环境、当前模型及训练数据保留。尚未完成的单元不能计入已释放空间。','','已有 AOSS 桶：`260010165`。归档版本：`'+p['archive_id']+'`。每个单元前缀见 `storage_plan.json`。','','恢复步骤见 `RESTORE.md`、`restore_archive.py`。清理执行状态见 `cleanup_controller_status.json`。','','平台个人配额接口未成功返回；容量按被清理文件的分配块计，最终目录占用检查另见 `release_cpu_result.json`。']
(HERE/'cleanup_progress.md').write_text('\n'.join(lines)+'\n');(HERE/'released_summary.json').write_text(json.dumps({'released_units':len(records),'allocated_bytes_removed':total,'units':records},indent=2)+'\n');print(json.dumps({'released_units':len(records),'GiB':total/2**30}));c.close()
