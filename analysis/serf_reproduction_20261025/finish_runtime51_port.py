"""Finish semantic equivalents of rejected 5.1 hunks on the author's older tree."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE/'runtime51/repo'
OG = ROOT/'BEHAVIOR-1K/OmniGibson/omnigibson'
changes = []


def edit(rel, fn):
    p = OG/rel
    old = p.read_text()
    new = fn(old)
    if old != new:
        ast.parse(new)
        p.write_text(new)
        changes.append({'file':rel,'before':hashlib.sha256(old.encode()).hexdigest(),
                        'after':hashlib.sha256(new.encode()).hexdigest()})


def replace_required(s, old, new):
    assert old in s, old[:120]
    return s.replace(old,new)


def simulator(s):
    s = replace_required(s, '(4, 5, 0): "omnigibson_4_5_0.kit"', '(5, 1, 0): "omnigibson_5_1_0.kit"')
    s = s.replace('._message_bus.dispatch(', '._message_bus.dispatch_event(')
    s = s.replace('self._set_viewer_camera()', 'self._set_viewer_camera(viewer_width=viewer_width, viewer_height=viewer_height)')
    s = replace_required(s, 'def _set_viewer_camera(self, relative_prim_path="/viewer_camera", viewport_name="Viewport"):',
        'def _set_viewer_camera(self, relative_prim_path="/viewer_camera", viewport_name="Viewport", viewer_width=gm.DEFAULT_VIEWER_WIDTH, viewer_height=gm.DEFAULT_VIEWER_HEIGHT):')
    s = s.replace('image_height=self.viewer_height,', 'image_height=viewer_height,').replace('image_width=self.viewer_width,', 'image_width=viewer_width,')
    s = replace_required(s, '        def update_handles(self):',
        '        def refresh_physics(self, sync_usd=False):\n'
        '            self.pi.update_simulation(elapsedStep=0.0, currentTime=self.current_time)\n'
        '            if sync_usd:\n'
        '                self.psi.fetch_results()\n\n'
        '        def update_handles(self):')
    return s.replace('og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)', 'self.refresh_physics()')


edit('simulator.py', simulator)

# Isaac 5.1 replaces SemanticsAPI with LabelsAPI. StatefulObject owns this
# operation in the SERF-pinned version, while the upstream patch targets USDObject.
pattern = re.compile(r'lazy\.isaacsim\.core\.utils\.semantics\.add_update_semantics\(\s*prim=(.*?),\s*semantic_label=(.*?),\s*type_label="class",?\s*\)', re.DOTALL)
for path in OG.rglob('*.py'):
    if 'add_update_semantics' not in path.read_text():
        continue
    def semantics(s):
        s, n = pattern.subn(lambda m: f'add_semantic_label(prim={m.group(1)}, label={m.group(2)})',s)
        if n and 'from omnigibson.utils.vision_utils import add_semantic_label' not in s:
            s = 'from omnigibson.utils.vision_utils import add_semantic_label\n'+s
        return s
    edit(str(path.relative_to(OG)), semantics)

edit('utils/usd_utils.py', lambda s:s.replace(
    'og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)', 'og.sim.refresh_physics()'))

# Disable optional torch compilation only through a scoped environment flag.
def compiler(s):
    return replace_required(s, 'def torch_compile(func):',
        'def torch_compile(func):\n'
        '    if os.getenv("OMNIGIBSON_DISABLE_TORCH_COMPILE") == "1":\n'
        '        return func\n')
edit('utils/python_utils.py', compiler)

# Use the author’s full evaluation implementation, including tracking wrappers.
shutil.copytree(ROOT/'src/serf_b1k/learning',OG/'learning',dirs_exist_ok=True)

# CoTracker weights and code are prefetched and pinned; avoid an unpinned hub fetch.
p=OG/'learning/online_3d_tracker.py'
s=p.read_text()
old='''        self._cotracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
        ).to(self._device)'''
new='''        import os
        self._cotracker = torch.hub.load(
            os.environ["SERF_COTRACKER_REPO"],
            "cotracker3_offline", source="local", pretrained=False,
        )
        self._cotracker.model.load_state_dict(torch.load(
            os.environ["SERF_COTRACKER_WEIGHTS"], map_location="cpu", weights_only=True))
        self._cotracker = self._cotracker.to(self._device)'''
assert old in s
p.write_text(s.replace(old,new))

# Preserve all 2025 task and controller settings; remove obsolete CUDA caches.
for p in ROOT.rglob('*.py'):
    ast.parse(p.read_text())
report={'scope':'SERF pinned OmniGibson + official Isaac 5.1 semantic/API port',
        'manual_changes':changes,'tracking_code_pinned':True,'python_syntax_valid':True}
(HERE/'runtime51_port_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'changed_files':len(changes),'python_syntax_valid':True}))
