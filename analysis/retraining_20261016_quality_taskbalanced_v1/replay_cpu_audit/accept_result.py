"""Independently validate the complete replay-window diagnostic result."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TERMINAL_SUCCESS = {"FINISHED", "SUCCESS", "SUCCEEDED"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_execution(native):
    if str(native['iid']) != '241735' or native['application_status'] not in TERMINAL_SUCCESS:
        raise ValueError('Native CPU job is not successfully terminal')
    if str(native.get('task_status', '')).lower() not in {'success', 'succeeded', 'finished'}:
        raise ValueError('Native task did not report success')
    workers = [c for c in native.get('containers', []) if c.get('role') == 'worker']
    if (len(workers) != 1 or workers[0].get('exitCode') not in (0, '0') or
            str(workers[0].get('status', '')).upper() not in TERMINAL_SUCCESS):
        raise ValueError('Data worker did not finish with exit code zero')
    return [c for c in native.get('containers', [])
            if c.get('role') != 'worker' and c.get('exitCode') not in (None, 0, '0')]


def check_window(row, expected, stats):
    for field, source in (("global_index", "global_index"), ("frame", "frame"),
                          ("episode_index", "episode_index"), ("task", "task_name")):
        if row[field] != expected[source]:
            raise ValueError("Window identity differs from the frozen request")
    dimension = row["largest_dimension"]
    if type(dimension) is not int or not 0 <= dimension < len(stats["mean"]):
        raise ValueError("Invalid action dimension")
    maximum, square = row["normalized_abs_max"], row["normalized_mean_square"]
    if not all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (maximum, square)):
        raise ValueError("Invalid normalized statistics")
    if square > maximum * maximum * (1 + 1e-7) + 1e-9:
        raise ValueError("Mean square exceeds the reported maximum")
    for name in ("raw_action_min_max_for_dimension", "delta_min_max_for_dimension"):
        bounds = row[name]
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError("Invalid action bounds")
    reconstructed = max(abs((value - stats["mean"][dimension]) / (stats["std"][dimension] + 1e-6))
                        for value in row["delta_min_max_for_dimension"])
    if not math.isclose(reconstructed, maximum, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("Reported target differs from the frozen normalization")


def main():
    if not (ROOT / "result.json").exists():
        print(json.dumps({"accepted": False, "reason": "result_not_fetched"}))
        return
    result = json.loads((ROOT / "result.json").read_text())
    proof = json.loads((ROOT / "fetch_proof.json").read_text())
    plan_path = ROOT / "bundle/column_plan.json"
    plan = json.loads(plan_path.read_text())
    norm_path = ROOT / "bundle/norm_stats.json"
    stats = json.loads(norm_path.read_text())["norm_stats"]["actions"]
    native, success = proof["native"], proof["success"]
    scheduler_observations = check_execution(native)
    if success.get("complete") is not True or success["result_sha256"] != sha(ROOT / "result.json"):
        raise ValueError("Result does not match the worker success receipt")
    if result.get("complete") is not True or any(result.get(k) is not False for k in ("training", "data_modified", "model_loaded")) or result.get("gpus") != 0:
        raise ValueError("Wrong diagnostic execution mode")
    if result["normalization_sha256"] != sha(norm_path) or result["column_plan_sha256"] != sha(plan_path):
        raise ValueError("Diagnostic input identity changed")
    expected = {w["global_index"]: w for item in plan["objects"] for w in item["windows"]}
    if len(expected) != plan["candidate_window_count"]:
        raise ValueError("Frozen request has duplicate windows")
    observed = {}
    for row in result["ranked_windows"]:
        index = row["global_index"]
        if index not in expected or index in observed:
            raise ValueError("Unknown or duplicate output window")
        check_window(row, expected[index], stats)
        observed[index] = row
    if set(observed) != set(expected) or result["windows"] != len(expected) or success["windows"] != len(expected):
        raise ValueError("Missing requested windows")
    if result["files"] != len(plan["objects"]) or success["files"] != len(plan["objects"]):
        raise ValueError("Missing requested Parquet files")
    if not 0 < result["range_bytes"] <= 48 * 1024**3 or success["range_bytes"] != result["range_bytes"]:
        raise ValueError("Invalid data transfer accounting")
    ranked = sorted(observed.values(), key=lambda x: x["normalized_abs_max"], reverse=True)
    report = {"accepted": True, "task_id": "241735", "windows": len(expected),
        "files": len(plan["objects"]), "range_bytes": result["range_bytes"],
        "result_sha256": sha(ROOT / "result.json"), "source_plan_sha256": sha(plan_path),
        "normalization_sha256": sha(norm_path), "top_windows": ranked[:20],
        "native_task_status": native['task_status'],
        "scheduler_exit_observations": scheduler_observations,
        "maximum_dimension_counts": dict(Counter(row["largest_dimension"] for row in ranked)),
        "above_50_std": sum(row["normalized_abs_max"] > 50 for row in ranked),
        "data_modified": False,
        "interpretation_limit": "Large normalized targets are candidates for diagnosis; this does not prove corrupt data or quantify model prediction error."}
    (ROOT / "accepted_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("accepted", "windows", "files", "above_50_std")}))


def self_test():
    expected = {"global_index": 1, "frame": 2, "episode_index": 3, "task_name": "test"}
    row = {"global_index": 1, "frame": 2, "episode_index": 3, "task": "test", "largest_dimension": 0,
           "normalized_abs_max": 1.0, "normalized_mean_square": 0.5,
           "raw_action_min_max_for_dimension": [0., 1.], "delta_min_max_for_dimension": [0., 1.000001]}
    stats = {"mean": [0.], "std": [1.]}
    check_window(row, expected, stats)
    for patch in ({"frame": 9}, {"normalized_abs_max": 2.}, {"normalized_mean_square": 2.}):
        try:
            check_window(dict(row, **patch), expected, stats)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid diagnostic row was accepted")
    native = {'iid': '241735', 'application_status': 'FINISHED', 'task_status': 'success',
              'containers': [{'role': 'worker', 'status': 'FINISHED', 'exitCode': 0},
                             {'role': 'scheduler', 'status': 'FAIL', 'exitCode': 255}]}
    assert len(check_execution(native)) == 1
    for patch in ({'task_status': 'failed'}, {'containers': [{'role': 'worker', 'status': 'FAIL', 'exitCode': 1}]},
                  {'containers': [{'role': 'scheduler', 'status': 'FINISHED', 'exitCode': 0}]}):
        try:
            check_execution(dict(native, **patch))
        except ValueError:
            pass
        else:
            raise AssertionError('Unsuccessful or missing data worker accepted')
    print(json.dumps({"valid_row_accepted": True, "wrong_identity_normalization_and_range_rejected": True,
                      "worker_success_required": True, "scheduler_nonzero_recorded_separately": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        self_test()
    else:
        main()
