"""Inspect frozen label exposure; labels are not verified physical progress."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'formal_stage/bundle'
sys.path.insert(0, str(BUNDLE / 'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def family(text):
    text = text.strip().lower()
    for prefix in ('move to ', 'pick up ', 'place ', 'open ', 'close ', 'hand over ', 'wipe ', 'toggle '):
        if text.startswith(prefix):
            return prefix.strip()
    return 'other'


def audit():
    plan = MixturePlan.load(BUNDLE / 'inputs/compiled/mixture.json')
    sources = {}
    for source in plan._phases[0]['sources']:
        exposure = Counter()
        counts = Counter()
        tasks = source['balanced_tasks']
        episodes_seen = 0
        long_move_episodes = 0
        for task, instances in tasks:
            for instance, episodes in instances:
                for identity, segments in episodes:
                    episodes_seen += 1
                    denominator = sum(s['end'] - s['start'] - plan.action_horizon + 1 for s in segments)
                    probability = 1 / (len(tasks) * len(instances) * len(episodes))
                    move = 0
                    for segment in segments:
                        n = segment['end'] - segment['start'] - plan.action_horizon + 1
                        key = family(segment['subtask'])
                        counts[key] += n
                        exposure[key] += probability * n / denominator
                        if key == 'move to':
                            move += n
                    if move / denominator > .8:
                        long_move_episodes += 1
        assert abs(sum(exposure.values()) - 1) < 1e-9
        sources[source['kind']] = dict(tasks=len(tasks), episodes=episodes_seen,
            legal_anchors=dict(counts), sampler_label_family_probability=dict(exposure),
            episodes_with_over80percent_move_to_label_anchors=long_move_episodes,
            language_loss_enabled=source['kind'] != 'full_episode')
    mixed = {}
    for name, weights in [('60:40', {'replay': .6, 'full_episode': .4}),
                          ('80:20', {'replay': .8, 'full_episode': .2})]:
        values = Counter()
        for kind, weight in weights.items():
            for key, probability in sources[kind]['sampler_label_family_probability'].items():
                values[key] += weight * probability
        assert abs(sum(values.values()) - 1) < 1e-9
        mixed[name] = dict(values)
    files = ['openpi/src/openpi/training/subtask_mixture_data.py', 'openpi/src/openpi/transforms.py',
             'inputs/compiled/mixture.json']
    result = dict(observed_at=time.time(),mixture_fingerprint=plan.fingerprint,
        sources=sources,mixed_label_family_probability=mixed,
        code_sha256={name:hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() for name in files},
        conditioning='Recorded weak subtask text remains action context; its CE loss mask is zero. Original labels retain CE.',
        limitation='move_to is a label prefix, not measured locomotion, inactivity, failure, or wasted frames; this audit cannot prove causality.',
        changes_to_training=0)
    (ROOT/'text_conditioning_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='code_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    audit()
