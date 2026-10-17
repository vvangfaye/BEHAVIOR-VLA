"""Build a local paired viewer only after source and downloaded hashes validate."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {19: '卸车', 11: '擦鞋', 18: '整理客厅'}
NOTES = {
    19: ['Q：0 → 1；原模型预算 566.2 秒耗尽，MSE 在 353.2 秒成功。',
         '约 210 秒：MSE 已从后备箱退开；约 270 秒进入关箱、继续搬运阶段。',
         '约 350 秒：MSE 已进入室内并在沙发附近放包；原模型仍在后备箱反复尝试取包，直到预算结束。',
         '提升集中在取包后的恢复与搬运链条，不能据此判断所有导航或抓取都更快。'],
    11: ['Q：0.5000 → 0.1667；两者均在 536.9 秒超时。',
         '约 120—240 秒：原模型有较长的鞋与刷子接触操作；MSE 的擦鞋阶段较短，之后转向放刷子。',
         'MSE 从约 322.7 秒进入放刷子阶段，原模型约 405.9 秒才首次进入；MSE 后续也长时间停留在床头柜附近。',
         '原生子任务标签中，擦鞋阶段累计占用时间为 74.7 → 18.7 秒。这是标签驻留时长，不能当作有效擦拭时间或清洁程度的测量。最终满足的顶层目标为 3 → 1。'],
    18: ['Q：0.2500 → 0；两者均在 628.3 秒超时。',
         '约 395—450 秒：原模型离开盆栽角落，走向茶几并放下报纸；MSE 仍持报纸在角落尝试抓盆栽。',
         '约 550 秒：原模型的报纸已在茶几上，并继续操作其他物品；MSE 仍停留在盆栽处，直到预算结束。',
         '抓盆栽标签累计占用约 96.0 → 417.1 秒。画面支持“长时间没有推进到后续步骤”，但不能直接确定是抓取控制、碰撞还是子任务切换造成。']}


def main():
    receipt = json.loads((ROOT / 'collection_receipt.json').read_text())
    assert receipt['complete'] and receipt['local_sha256_verified']
    selection = json.loads((ROOT / 'selection.json').read_text())
    for row in selection['files']:
        p = ROOT / 'media' / row['relative_path']
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    sections = []
    for index in selection['instances']:
        cells, steps = [], []
        for role, label in [('baseline', '原模型'), ('candidate', '纯专家 MSE')]:
            identity = f'{role}-{index:02d}'
            report = json.loads((ROOT.parent / 'reports' / (identity + '.json')).read_text())
            row = report['record']
            video = next(f for f in selection['files'] if f['id'] == identity and f['path'].endswith('.mp4'))
            relative = 'media/' + video['relative_path']
            steps.append(row['steps'])
            cells.append('<div><p><b>' + label + '</b> · Q=' + f"{row['q_score']:.4f}" +
                         ' · ' + ('成功' if row['model_success'] else '未成功') +
                         ' · ' + str(row['steps']) + ' 步</p><video controls preload="metadata" src="' +
                         html.escape(relative, quote=True) + '"></video><p><a href="' +
                         html.escape(relative, quote=True) + '">打开原视频</a></p></div>')
        segments = json.loads((ROOT / 'highlights' / f'pair-{index:02d}-segments.json').read_text())
        windows = '、'.join(f'{a}—{b} 秒' for a, b in segments['simulation_time_windows_seconds'])
        notes = '<ul>' + ''.join('<li>' + html.escape(n) + '</li>' for n in NOTES[index]) + '</ul>'
        highlight = '<p>精选片段：左原模型，右纯专家 MSE，4 倍速；依次拼接仿真时间 ' + windows + '。</p><video class="highlight" controls preload="metadata" src="highlights/pair-' + f'{index:02d}' + '-highlight.mp4"></video>'
        sections.append('<section><h2>' + LABELS[index] + '</h2>' + notes + highlight + '<h3>完整视频</h3><div class="pair">' + ''.join(cells) +
                        '</div><button onclick="playPair(this)">同时播放</button> '
                        '<button onclick="pausePair(this)">暂停</button> '
                        '<label>倍速 <select onchange="speedPair(this)"><option>1</option><option>4</option>'
                        '<option selected>8</option><option>16</option></select></label> '
                        '<label>定位到仿真时间（秒）<input type="number" min="0" value="0" '
                        'onchange="seekPair(this)"></label></section>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原模型与纯专家 MSE 视频对比</title>
<style>body{font:16px system-ui;color:#192330;max-width:1500px;margin:24px auto;padding:0 20px;background:#f7f9fc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#111}section{background:white;padding:18px;margin:20px 0;border:1px solid #d7dfe8;border-radius:10px}button,select,input{font:inherit;padding:6px}input{width:95px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<h1>原模型与纯专家 MSE：视频对比</h1><p>来自正在进行的独立验证，仅选取已完成配对中 Q 差异最大的一个提升案例与两个退步案例。用于分析行为差异，不代表整体成绩。</p>
<p>两侧按相同仿真时间（视频 30 帧/秒）对齐；较早结束的一侧停在末帧。默认 8 倍速，可分别查看原始视频。画面仅能支持可见行为判断，不能单独证明内部推理原因。</p>'''+''.join(sections)+'''
<script>
function videos(e){return [...e.closest('section').querySelectorAll('.pair video')]}
function playPair(e){let vs=videos(e),rate=+e.closest('section').querySelector('select').value;vs.forEach(v=>{v.playbackRate=rate;v.play().catch(()=>{})})}
function pausePair(e){videos(e).forEach(v=>v.pause())}
function speedPair(e){videos(e).forEach(v=>v.playbackRate=+e.value)}
function seekPair(e){videos(e).forEach(v=>v.currentTime=Math.min(+e.value,Number.isFinite(v.duration)?v.duration:+e.value))}
</script></html>'''
    (ROOT / 'compare.html').write_text(document)
    print(json.dumps({'viewer': str(ROOT / 'compare.html'), 'paired_tasks': 3, 'verified_files': len(selection['files'])}))


if __name__ == '__main__':
    main()
