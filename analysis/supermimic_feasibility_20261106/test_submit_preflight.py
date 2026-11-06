import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("export_submit", Path(__file__).with_name("submit_export.py"))
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def test_explicit_empty_and_structured_job_lists():
    assert submit.parse_job_list("No jobs found\n") == []
    assert submit.parse_job_list("[]") == []
    assert submit.parse_job_list('{"training_jobs": [{"uid": "known"}]}') == [{"uid": "known"}]


@pytest.mark.parametrize("text", ["", "unrecognized", "{}"])
def test_unknown_preflight_output_is_not_treated_as_empty(text):
    with pytest.raises(ValueError):
        submit.parse_job_list(text)
