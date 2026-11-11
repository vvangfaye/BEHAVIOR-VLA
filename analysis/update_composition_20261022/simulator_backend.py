"""Load the existing evaluator bridge with a bounded cold-start allowance."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path

_clients = {}


@contextmanager
def simulator(plan, output, instance, mode, max_steps):
    if 'renderer_mailbox' in plan:
        from renderer_mailbox import Client, RemoteSimulator
        config=plan['renderer_mailbox']
        key=(config['directory'],config['session'])
        client=_clients.setdefault(key,Client(*key))
        bridge=RemoteSimulator(client,plan,output,instance,mode,max_steps)
        try:
            yield bridge
        finally:
            bridge.close()
        return
    source=Path(plan['simulator_common'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['simulator_common_sha256']:
        raise ValueError('Simulator bridge source differs from the frozen plan')
    spec=importlib.util.spec_from_file_location('composition_simulator_bridge',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Bridge(module.Simulator):
        def receive(self,timeout=180):
            # The previous native adapter check timed out during cold kernel
            # compilation at 600 seconds; retain a finite documented bound.
            if timeout==600:timeout=plan['startup_timeout_seconds']
            return super().receive(timeout)
    bridge=Bridge.__new__(Bridge)
    try:
        bridge.__init__(plan,output,instance,mode,max_steps)
        yield bridge
    finally:
        # Also clean up a subprocess if initialization fails before yield.
        if hasattr(bridge,'process'):
            bridge.close()


def shutdown_renderer(plan):
    config=plan.get('renderer_mailbox')
    if config is not None:
        client=_clients.get((config['directory'],config['session']))
        if client is not None:
            client.request(dict(op='shutdown'),timeout=120)
