"""Build a review artifact from the six preserved episodes, without rerunning them."""
from pathlib import Path
import ast
import difflib
import html
import json
import time

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
comparison = json.loads((ROOT / "comparison.json").read_text())
timeline = json.loads((HERE / "timeline.json").read_text())
alignment = json.loads((HERE / "map_alignment.json").read_text())
protocol = json.loads((HERE / "protocol_retry_test.json").read_text())

cases = []
for row in comparison["rows"]:
    configs = {}
    for condition in ("image", "serf"):
        source = Path(row[condition]["source"])
        case = source.parent
        configs[condition] = yaml.safe_load((case / "config.yaml").read_text())
        action_dir = next(case.glob("repro/actions/*/*/*/manifest.json")).parent
        actions = np.concatenate([np.load(f) for f in sorted(action_dir.glob("chunk_*.npy"))])
        assert actions.shape == (15385, 23) and np.isfinite(actions).all()
        t = next(v for v in timeline if v["instance"] == row["instance"] and v["condition"] == condition)
        cases.append({
            "instance": row["instance"], "condition": condition,
            "score": row[condition]["score"], "success": row[condition]["success"],
            "action_shape": list(actions.shape), "all_actions_finite": True,
            "last_score_change_step": t["score_changes"][-1]["step"],
            "peak_score": t["peak_score"],
            "video": str(next((case / "videos").glob("*.mp4")).relative_to(ROOT)),
        })
    for key in ("eval_instance_ids", "replay_mode", "max_steps", "robot", "task", "model",
                "partial_scene_load", "test_hidden"):
        assert configs["image"][key] == configs["serf"][key], (row["instance"], key)
    assert row["image"]["initial"]["initial_state_sha256"] == row["serf"]["initial"]["initial_state_sha256"]

# Stage a transport fix for review. Preserve the source used for the six episodes.
network_file = ROOT / "runtime51/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/network_utils.py"
before = network_file.read_text()
after = before.replace(
    'self._uri, compression=None, max_size=None, additional_headers=headers',
    'self._uri, compression=None, max_size=None, additional_headers=headers, ping_interval=None',
)
old_retry = '''            except websockets.exceptions.ConnectionClosedError:
                logger.warning("Connection to server lost, attempting to reconnect...")
                self._ws, self._server_metadata = self._wait_for_server()'''
new_retry = '''            except websockets.exceptions.ConnectionClosedError as exc:
                # A stateful policy may already have advanced its action queue.
                # Retrying the observation can silently skip an action.
                raise RuntimeError(
                    "Connection lost during stateful inference; abort the episode "
                    "instead of replaying an ambiguous request."
                ) from exc'''
assert old_retry in after
after = after.replace(old_retry, new_retry)
after = after.replace('            process_request=_health_check,',
                      '            process_request=_health_check,\n            ping_interval=None,')
ast.parse(after)
(HERE / "network_utils_candidate.py").write_text(after)
(HERE / "transport_fix.patch").write_text("".join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True),
    fromfile="a/omnigibson/learning/utils/network_utils.py",
    tofile="b/omnigibson/learning/utils/network_utils.py",
)))

observations = {
    73: "SERF 在约 6500 步把棕色凉鞋落在架外地面；7192 步后 score 一直停在 0.3，后半程离开鞋架。图像基线随后继续搬放，峰值 0.7，最终 0.6。",
    256: "SERF 先把棕色凉鞋放到上面的平板，在下层放运动鞋时反复调整；6147 步 score 降至 0.2，之后不再上升。图像基线后续继续搬放，最终 0.5。",
    282: "SERF 先后搬运两只凉鞋、两只运动鞋；10563 步达到 0.8。图像基线大部分时间用在抓取和重复调整，15205 步才首次达到 0.4。最终未满足全部目标谓词。",
}
report = {
    "updated_at_unix": time.time(), "scope": "Read-only audit of six existing episodes; no GPU job submitted",
    "cases": cases,
    "checks_passed": [
        "Per-instance serialized initial states match; physical reload delta below 1e-4",
        "Task, instances, action/controller definitions, replay setting and max-step policy agree",
        "All six episodes executed 15385 finite 23-dimensional actions",
        "Timeout follows author's 2x mean human-demo length; normalized_time=0.5 is a reciprocal metric, not half an episode",
        "Published map features and positions are finite; target centroids lie 0.015-0.025 m from corresponding initial object origins",
        "480px SERF capture, 224px image capture, common head aperture and 224px policy resize follow author code",
    ],
    "confirmed_transport_defect": {
        "observed_in_all_six_logs": "Startup keepalive timeout followed by reconnection",
        "reproduction": protocol,
        "consequence": "A timed-out request may consume the first stateful action, and the retry consumes the second; may offset SERF map refresh boundaries",
        "causal_limit": "The exact skipped-action count in the preserved episodes was not recorded; this shared defect does not by itself explain which model scores higher",
        "fix_status": "Candidate patch staged and syntax checked; frozen evaluation source and all six results unchanged",
    },
    "unresolved": [
        "No per-boundary tracking IDs, point-cloud residuals or transforms were saved; dynamic map correctness cannot be certified from RGB and warning counts",
        "Missing target category in current visible segmentation can be normal occlusion; warning frequency alone is not evidence of a broken tracker",
        "Isaac 4.5-to-5.1 physics/rendering equivalence has not been established",
        "Final per-predicate truth values were not saved; the exact missing predicates in instance 282 are unknown",
    ],
    "observations": observations,
    "conclusion": "The six RGB rollouts are real, but the protocol defect prevents treating this run as a clean test of SERF memory benefit. Fix and verify transport, then inspect tracking quality before assigning causality.",
}
(HERE / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

# Self-contained local player with paired seek and original, unmodified videos.
payload = []
for row in comparison["rows"]:
    n = row["instance"]
    pair = {"instance": n, "observation": observations[n], "conditions": {}}
    for condition in ("image", "serf"):
        case = next(v for v in cases if v["instance"] == n and v["condition"] == condition)
        t = next(v for v in timeline if v["instance"] == n and v["condition"] == condition)
        pair["conditions"][condition] = dict(case, url="../" + case["video"], changes=t["score_changes"])
    payload.append(pair)
page = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SERF 三实例回放核查</title>
<style>body{max-width:1450px;margin:28px auto;padding:0 20px;font:16px system-ui;color:#263445;background:#f5f6f8}h1{font-size:25px}button,select,input{font:inherit;padding:6px}button{cursor:pointer}section{background:white;padding:18px;margin:18px 0;border-radius:9px}.warn{border-left:4px solid #c27b11;padding:13px;background:#fff7e5}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}video,img{width:100%;background:#222}.controls{display:flex;align-items:center;gap:10px;margin:14px 0}input[type=range]{flex:1}svg{width:100%;height:120px}.muted{color:#596777}</style>
<h1>SERF：73、256、282 的配对回放与评测核查</h1>
<p class="warn">已发现首次推理超时后的自动重试问题。下面是真实旧回放，不是修复后的新实验；不能据此给“记忆有效/无效”作因果结论。</p>
<p>所有视频均为 30 FPS；时间为仿真时间。score 表示新满足的目标谓词比例，阶段预测不等于任务实际完成。</p>
<label>实例 <select id="choose"></select></label><section><p id="observation"></p>
<div class="pair"><div><h3 id="aTitle"></h3><video id="a" controls preload="metadata"></video></div><div><h3 id="bTitle"></h3><video id="b" controls preload="metadata"></video></div></div>
<div class="controls"><button id="toggle">同步播放 / 暂停</button><select id="speed"><option>1</option><option>2</option><option selected>4</option><option>8</option></select><input id="seek" type="range" min="0" max="512.8" step="0.1"><span id="time"></span></div><div id="jumps"></div><svg id="plot" viewBox="0 0 1000 130"></svg>
<p class="muted">蓝线：图像基线；橙线：SERF。下方关键帧均截自原始回放。</p><img id="frames"></section>
<section><h3>已核对</h3><p>相同初态、任务、控制定义、步数预算；发布地图中四只鞋的初始位置对应正确。SERF 的高分辨率采集与图像模型的低分辨率采集来自作者各自的默认配置。</p><h3>尚未排除</h3><p>动态地图的跟踪误差未记录；Isaac 5.1 的物理等价性未验证。物体 ID 警告可能只是不可见，不能按警告次数推断跟踪失败。</p><p><a href="audit.json">核查记录</a> · <a href="protocol_retry_test.json">本机 CPU 协议复现结果</a> · <a href="transport_fix.patch">待应用的协议补丁</a></p></section>
<script>const DATA=__DATA__;const $=x=>document.getElementById(x);let row;const vs=[$('a'),$('b')];let seeking=false;
for(const d of DATA)$('choose').add(new Option(d.instance,d.instance));
function jump(t){for(const v of vs)v.currentTime=t;$('seek').value=t;label(t)}
function label(t){$('time').textContent=t.toFixed(1)+' s / step '+Math.round(t*30)}
function scorePath(changes){let q=0,s='M 0 110';for(const p of changes){const x=p.step/15385*1000;s+=' H '+x.toFixed(2)+' V '+(110-p.score*100).toFixed(2);q=p.score}return s+' H 1000'}
function load(){vs.forEach(v=>v.pause());row=DATA.find(x=>x.instance===$('choose').value*1);$('observation').textContent=row.observation;
['image','serf'].forEach((c,i)=>{vs[i].src=row.conditions[c].url;vs[i].playbackRate=+$('speed').value;$(i?'bTitle':'aTitle').textContent=(i?'SERF':'图像基线')+'，最终 score '+row.conditions[c].score+'，完整成功：否'});
$('frames').src=row.instance+'_serf_keyframes.jpg';$('jumps').replaceChildren();for(const t of [0,100,150,200,216.7,240,300,350,450,510]){const b=document.createElement('button');b.textContent=t+'s';b.onclick=()=>jump(t);$('jumps').append(b)}
$('plot').innerHTML='<path d="M0 10V110H1000" fill="none" stroke="#ccd3dc"/>'+['image','serf'].map((c,i)=>'<path d="'+scorePath(row.conditions[c].changes)+'" fill="none" stroke="'+(i?'#c8751f':'#2573b7')+'" stroke-width="2"/>').join('');label(0)}
$('choose').onchange=load;$('seek').oninput=()=>jump(+$('seek').value);$('speed').onchange=()=>vs.forEach(v=>v.playbackRate=+$('speed').value);
$('toggle').onclick=()=>{if(vs[0].paused){vs.forEach(v=>v.play().catch(()=>{}))}else vs.forEach(v=>v.pause())};
vs[0].ontimeupdate=()=>{const t=vs[0].currentTime;$('seek').value=t;label(t);if(!vs[0].paused&&Math.abs(vs[1].currentTime-t)>.6)vs[1].currentTime=t};load();</script></html>'''
(HERE / "replay_review.html").write_text(page.replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
print(json.dumps({"checked_episodes": len(cases), "report": str(HERE / "replay_review.html"), "transport_fix_applied": False}))
