"""Run the unchanged B1K server with an explicit repeat-seed protocol."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
    from full_episode_repeat_seeding import install_server_protocol
    from openpi.serving import websocket_b1k_server
    from openpi.shared import eval_b1k_wrapper
    install_server_protocol(websocket_b1k_server, eval_b1k_wrapper)
    runpy.run_path(str(Path(__file__).with_name('serve_b1k.py')), run_name='__main__')
