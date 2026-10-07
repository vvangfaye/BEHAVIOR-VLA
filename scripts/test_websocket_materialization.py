"""Real WebSocket regression for long simulator restoration and first inference."""
import asyncio
import importlib.util
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import numpy as np
import pytest
th=pytest.importorskip('torch')
import requests
import websockets
import websockets.asyncio.server as ws_server
import websockets.sync.client
from test_behavior_version_compat import load_definitions, EVAL

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('actual_b1k_server',ROOT/'openpi/src/openpi/serving/websocket_b1k_server.py')
server_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(server_module)
Client=load_definitions(EVAL/'utils/network_utils.py',['WebsocketClientPolicy'],Packer=server_module.Packer,
    unpackb=server_module.unpackb,deepcopy=server_module.deepcopy,logger=logging.getLogger(__name__),
    websockets=websockets,requests=requests,time=time,urlparse=urlparse,
    ACTION_CHUNK_REQUEST_KEY='__action_chunk_size__').WebsocketClientPolicy


class Policy:
    def __init__(self,slow=False):self.resets=0;self.inferences=0;self.slow=slow
    def reset(self):self.resets+=1
    def act(self,obs):
        self.inferences+=1
        if self.slow:time.sleep(.2)
        return th.from_numpy(np.arange(23,dtype=np.float32).reshape(1,23))


@pytest.fixture
def live_server(monkeypatch):
    """Run the actual production server.run and handler with an ephemeral port."""
    started=[];original=ws_server.serve
    def launch(*,old_deadline=False,slow=False):
        ready=threading.Event();state={};policy=Policy(slow)
        class Context:
            def __init__(self,*args,**kw):
                if old_deadline:kw.update(ping_interval=.025,ping_timeout=.025)
                else:assert kw['ping_interval'] is None
                self.inner=original(*args,**kw)
            async def __aenter__(self):
                server=await self.inner.__aenter__()
                state.update(port=server.sockets[0].getsockname()[1],loop=asyncio.get_running_loop(),task=asyncio.current_task())
                ready.set();return server
            async def __aexit__(self,*args):return await self.inner.__aexit__(*args)
        # Change only this source module's server factory, not the library globals.
        monkeypatch.setattr(server_module,'_server',SimpleNamespace(serve=Context))
        server=server_module.WebsocketPolicyServer(policy,host='127.0.0.1',port=0,server_token='local-regression')
        errors=[]
        def target():
            try:asyncio.run(server.run())
            except asyncio.CancelledError:pass
            except BaseException as e:errors.append(e);ready.set()
        thread=threading.Thread(target=target,daemon=True);thread.start()
        assert ready.wait(5) and not errors
        client=Client(host='127.0.0.1',port=state['port'],expected_server_token='local-regression',server_wait_timeout=5)
        client._ws,client._server_metadata=client._wait_for_server()
        started.append((client,state,thread,errors))
        return client,policy
    yield launch
    for client,state,thread,errors in reversed(started):
        client._ws.close()
        if thread.is_alive():state['loop'].call_soon_threadsafe(state['task'].cancel)
        thread.join(5)
        assert not thread.is_alive() and not errors


def ignore_heartbeat_pings(client):
    """Emulate a simulator process unable to answer PING during restoration."""
    original=client._ws.protocol.recv_frame
    def recv(frame):
        if frame.opcode==websockets.frames.Opcode.PING:return
        original(frame)
    client._ws.protocol.recv_frame=recv


def test_old_heartbeat_deadline_reproduces_reset_failure_after_restoration(live_server):
    client,policy=live_server(old_deadline=True)
    ignore_heartbeat_pings(client);time.sleep(.2)
    assert policy.inferences==0
    with pytest.raises(websockets.exceptions.ConnectionClosedError,match='keepalive ping timeout'):client.reset()


def test_actual_fixed_connection_survives_restore_then_reset_and_action(live_server):
    client,policy=live_server();ignore_heartbeat_pings(client);time.sleep(.2)
    client.reset();action=client.act({'proprio':np.zeros(4,np.float32)})
    assert policy.resets==1 and policy.inferences==1
    assert th.equal(action,th.arange(23,dtype=th.float32).reshape(1,23))
    assert client.get_server_metadata()['server_instance_token']=='local-regression'


def test_actual_fixed_connection_survives_blocking_first_inference_and_next_reset(live_server):
    client,policy=live_server(slow=True)
    first=client.act({});client.reset();second=client.act({})
    assert policy.resets==1 and policy.inferences==2 and th.equal(first,second)
