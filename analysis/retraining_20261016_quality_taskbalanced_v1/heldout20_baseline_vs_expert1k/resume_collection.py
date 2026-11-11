"""Resume collection only: no job creation, staging, stop, or episode retry."""
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_holdout_collector', ROOT / 'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
original_observe = controller.observe


def existing_group(plan, group):
    path = ROOT / 'intents' / (group['id'] + '.json')
    intent = controller.load(path)
    if (not intent.get('uid') or intent['name'] != group['name']
            or intent['members'] != group['members']):
        raise ValueError('Existing group identity differs; collector cannot submit')


def existing_stage(client, plan):
    if not (ROOT / 'intents/stage.json').exists() or not (ROOT / 'stage_accepted.json').exists():
        raise ValueError('Existing accepted stage required; collector cannot stage')


def prohibit_creation(*args, **kwargs):
    raise ValueError('Collection-only recovery cannot create any cloud job')


def observe(intent):
    for attempt in range(4):
        try:
            return original_observe(intent)
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            with (ROOT / 'collection_query_retries.jsonl').open('a') as stream:
                stream.write(json.dumps({'at': time.time(), 'name': intent['name'],
                                        'uid': intent['uid'], 'attempt': attempt + 1,
                                        'category': 'native_read_query_failed'}) + '\n')
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    plan = controller.load(ROOT / 'plan.json')
    existing_stage(None, plan)
    for group in plan['groups']:
        existing_group(plan, group)
    controller.tracked_intents(plan)  # Also validate the separately admitted recovery.
    controller.prepare_stage = existing_stage
    controller.submit_group = existing_group
    controller.native_create = prohibit_creation
    controller.observe = observe
    while time.time() < plan['deadline_epoch'] and not (ROOT / 'STOP').exists():
        try:
            controller.main()
            return
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            controller.save(ROOT / 'collection_observation_error.json', {
                'at': time.time(), 'category': 'native_read_query_failed_after_retries',
                'jobs_created': 0, 'episodes_restarted': 0})
            time.sleep(60)


if __name__ == '__main__':
    main()
