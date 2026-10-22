"""A bounded RTX simulation worker for a single H100 training process."""
import argparse
import json
from pathlib import Path
import time

from renderer_mailbox import decode, publish
from simulator_backend import simulator


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def serve(plan, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = plan['environment']['renderer_mailbox']
    root = Path(config['directory'])
    root.mkdir(parents=True, exist_ok=True)
    session = config['session']
    status_path = root / 'status.json'
    if status_path.exists():
        raise FileExistsError('Renderer session directory already used')
    environment = dict(plan['environment'])
    environment.pop('renderer_mailbox')
    environment['simulator_common'] = str(Path(__file__).with_name('common.py'))
    allowed = {(row['task'], instance, seed, row['max_steps'])
               for row in plan['iterations']
               for instance, seed in zip(row['instances'], row['environment_seeds'])}
    sequence = 0
    active = None
    context = None
    episodes = 0
    started = last_activity = time.monotonic()
    deadline = started + plan['max_seconds']
    failure = None
    prewarmed_request = None
    def status(stage, terminal=False):
        value = dict(session=session, stage=stage, terminal=terminal,
                     sequence=sequence, episodes=episodes,
                     elapsed_seconds=time.monotonic() - started)
        write(status_path, value)
        write(output/'progress.json', value)
    def open_episode(command):
        nonlocal active,context,episodes
        if active is not None:
            raise ValueError('An episode is already open')
        if episodes >= plan['max_training_episodes']:
            raise ValueError('The declared episode budget is exhausted')
        key = (command['task'], command['instance'], command['seed'], command['max_steps'])
        if key not in allowed or command['mode'] != 'train':
            raise ValueError('Episode is not in the frozen training plan')
        target = Path(command['output']).resolve()
        if not target.is_relative_to(Path(config['episode_output_root']).resolve()):
            raise ValueError('Episode output escapes the declared output root')
        env = dict(environment, task=command['task'], seed=command['seed'])
        candidate = simulator(env,target,command['instance'],command['mode'],command['max_steps'])
        active = candidate.__enter__()
        context = candidate
        episodes += 1
    try:
        status('ready')
        if config.get('prewarm_first',False):
            first=plan['iterations'][0]
            prewarmed_request=dict(op='open',task=first['task'],instance=first['instances'][0],
                seed=first['environment_seeds'][0],max_steps=first['max_steps'],mode='train',
                output=str(Path(config['episode_output_root'])/'iteration_000_episode_0'))
            status('prewarming')
            open_episode(prewarmed_request)
            last_activity=time.monotonic()
            status('prewarmed')
        while time.monotonic() < deadline:
            request_path = root / f'{sequence:08d}.request.npz'
            if not request_path.exists():
                if time.monotonic() - last_activity > config['idle_timeout_seconds']:
                    raise TimeoutError('Training process did not send a request before the idle limit')
                time.sleep(.1)
                continue
            request = decode(request_path.read_bytes())
            if request['session'] != session or request['sequence'] != sequence:
                raise ValueError('Renderer request identity mismatch')
            command = request['command']
            operation = command['op']
            status(operation)
            stopping = False
            if operation == 'open':
                if prewarmed_request is not None:
                    if command!=prewarmed_request:
                        raise ValueError('First request differs from the prewarmed episode')
                    prewarmed_request=None
                else:
                    open_episode(command)
                result = active.current
            elif operation == 'step':
                if active is None:
                    raise ValueError('No open episode')
                result = active.step(command['actions'], command['subtask'])
            elif operation == 'close':
                if context is not None:
                    context.__exit__(None, None, None)
                active = context = None
                result = {'closed': True}
            elif operation == 'shutdown':
                if active is not None:
                    raise ValueError('Close the episode before shutdown')
                result = {'closed': True}
                stopping = True
            else:
                raise ValueError('Unknown renderer operation')
            publish(root/f'{sequence:08d}.response.npz',
                    dict(session=session, sequence=sequence, result=result))
            sequence += 1
            last_activity = time.monotonic()
            if stopping:
                break
        else:
            raise TimeoutError('Renderer wall-time limit reached')
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'[:600]
        # Publish a failure to the waiting trainer whenever no response exists.
        destination = root/f'{sequence:08d}.response.npz'
        if (root/f'{sequence:08d}.request.npz').exists() and not destination.exists():
            publish(destination, dict(session=session, sequence=sequence, error=failure))
        raise
    finally:
        try:
            if context is not None:
                context.__exit__(None, None, None)
        finally:
            status('failed' if failure else 'complete', terminal=True)
            write(output/'result.json', dict(complete=failure is None, error=failure,
                rendered_episodes=episodes, training_episodes=0,
                scope='Simulation service only; policy and training live on H100'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    serve(json.loads(args.plan.read_text()), args.output)
