import json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
state=json.loads((HERE/'cleanup_controller_status.json').read_text());assert state['stage']=='complete',state
p=json.loads((HERE/'storage_plan.json').read_text());r=json.loads((HERE/'released_summary.json').read_text());v=json.loads((HERE/'release_cpu_result.json').read_text())
assert r['released_units']==len(p['units'])==10
assert v['write_probe_succeeded'] is True
assert set(v['remaining_files'])=={u['id'] for u in p['units']}
assert all(sorted(files)==['AFS_ARCHIVED.json'] for files in v['remaining_files'].values()),v['remaining_files']
assert sum(u['released_files'] for u in r['units'])==sum(len(u['files']) for u in p['units'])
proof={'archive_id':p['archive_id'],'verified_at':time.time(),'units':r['released_units'],'files_released':sum(u['released_files'] for u in r['units']),'allocated_file_bytes_removed':r['allocated_bytes_removed'],'removed_gib':r['allocated_bytes_removed']/2**30,'final_batch_directory_bytes_reduced':v['directory_bytes_reduced'],'all_source_directories_only_recovery_pointer':True,'write_probe_succeeded':True,'all_temporary_cpu_apps_stopped_by_controller':True,'personal_quota_api_verified':False}
(HERE/'final_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
lines=['# AFS 清理完成（2026-10-23）','',f'本轮清理 **{proof["removed_gib"]:.2f} GiB**，共 {proof["files_released"]} 个文件、10 个独立单元。','', '全部待清理内容均已完成 HTTPS 源站全量回读和 SHA256 校验，清理前重新检查活动依赖及源内容。所有源目录现在只留下 `AFS_ARCHIVED.json` 恢复指针，最终写入探针成功。','', '| 已清理源目录 | GiB |','|---|---:|']
for u in r['units']:lines.append(f'| `{u["source"]}` | {u["unlinked_allocated_bytes"]/2**30:.2f} |')
lines+=['','## 保留内容','','当前运行环境、当前模型和训练数据未清理；用途未确认的约60 GiB训练视频及原始示范保留。评测报告、视频和代码快照保留。','','## 恢复入口','','- AOSS 桶：`260010165`','- 版本：`'+p['archive_id']+'`','- 每个单元前缀和文件清单见 `storage_plan.json` 与归档内 `MANIFEST.json`。','- 使用 `restore_archive.py --prefix <单元前缀> --destination <新目录>`；先检查个人容量预算。','- `existing_object_key` 指向的既有对象仍是恢复依赖，不可把相应旧前缀当作垃圾删除。','','## 容量与资源核验','','最终批次同时做了清理前后 `du` 及写入检查。累计 GiB 为被删除文件的分配块之和；个人配额接口未成功返回，因此不声称全卷剩余量就是个人可用额度。临时CPU应用已停止；未新增GPU资源。']
(HERE/'cleanup_progress.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(proof))
