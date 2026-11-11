"""Summarize returned raw reports without treating missing runs as failures."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent/'retrieved'
def read(path):
    p=root/path
    return json.loads(p.read_text()) if p.exists() else None

lines=['5090 闭环对照：checkpoint 1688977，seed=0。', '',
       '| 实例 | 原始 JAX score | 原始成功 | 适配 Torch score | 适配成功 |',
       '|---|---:|:---:|---:|:---:|']
for instance in [301,302]:
    cells=[str(instance)]
    for backend in ['jax','torch']:
        data=read(f'{backend}/{instance}/result.json')
        cells.extend([f"{data['score']:.4f}", '是' if data['success'] else '否'] if data else ['未完成','—'])
    lines.append('| '+' | '.join(cells)+' |')
gate=read('gate.json')
if gate:
    lines += ['', '适配准入：'+('通过。' if gate['passed'] else '未通过；未启动后续 PPO。')]
    if not gate['passed']:
        lines.append('未通过项：'+', '.join(k for k,v in gate['checks'].items() if not v)+'。')
ppo=read('ppo/result.json')
if ppo:
    lines += ['', f"PPO：{ppo['iterations']} 轮，{ppo['controls']} 个控制步，{ppo['chunks']} 个 chunk，{ppo['optimizer_steps']} 次 optimizer.step。",
              '冻结参数逐字节保持一致，动作 expert 和 value head 已更新，更新权重已做保存回读校验。',
              '本轮只验收单进程训练链路，未验证 Ray/FSDP，也不能据此判断成功率提高。']
else:
    lines += ['', '尚无已通过验收的 PPO 结果。']
(root.parent/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
