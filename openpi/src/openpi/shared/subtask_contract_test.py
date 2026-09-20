import pytest

from openpi.shared import subtask_contract


def test_checkpoint_conditioning_is_persisted_and_cannot_change_on_resume(tmp_path):
    assert subtask_contract.load(tmp_path) is None
    subtask_contract.check_resume(tmp_path, "task_and_subtask")
    with pytest.raises(ValueError, match="new fine-tuning experiment"):
        subtask_contract.check_resume(tmp_path, "subtask_only")
    subtask_contract.save(tmp_path, "subtask_only")
    assert subtask_contract.load(tmp_path) == "subtask_only"
    subtask_contract.check_resume(tmp_path, "subtask_only")
    with pytest.raises(ValueError, match="Cannot resume"):
        subtask_contract.check_resume(tmp_path, "task_and_subtask")


def test_invalid_metadata_is_not_silently_treated_as_legacy(tmp_path):
    (tmp_path / subtask_contract.FILENAME).write_text('{"version": 2, "action_conditioning": "subtask_only"}')
    with pytest.raises(ValueError, match="Invalid joint"):
        subtask_contract.load(tmp_path)
