"""Native policy, with an explicit seed application/acknowledgement adapter."""
import os
import runpy
from seed_protocol import install_server_protocol
from openpi.serving import websocket_b1k_server
from openpi.shared import eval_b1k_wrapper

install_server_protocol(websocket_b1k_server, eval_b1k_wrapper)
runpy.run_path(os.environ['PAIRED_NATIVE_SERVER'], run_name='__main__')
