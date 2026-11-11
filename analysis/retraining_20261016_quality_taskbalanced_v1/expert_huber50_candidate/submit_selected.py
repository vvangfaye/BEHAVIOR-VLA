"""Use the same checked submission implementation for the Huber 50 variant."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "expert_huber_candidate/submit_selected.py"
spec = importlib.util.spec_from_file_location("huber50_shared_submit", SOURCE)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.HERE = HERE
shared.STAGE = HERE / "selected_stage"


def validate_local():
    return shared.validate_local()


if __name__ == "__main__":
    shared.main()
