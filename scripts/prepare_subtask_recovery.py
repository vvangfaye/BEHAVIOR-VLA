#!/usr/bin/env python3
"""Prepare immutable valid-interval indices without reading/reencoding videos."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
from openpi.training.recovery_index import canonical_digest, file_sha256, prepare_index, verify_source_files


def held_out_from_file(path):
    """Accept a direct split list or one consistently frozen full299 campaign."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list): return payload
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list) or not payload["groups"]:
        raise ValueError("Held-out input must be a split list or a campaign with groups")
    normalized = None
    for group in payload["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("plan"), dict):
            raise ValueError("Campaign group lacks a frozen plan")
        held = group["plan"].get("held_out_instances")
        if not isinstance(held, list) or any(not isinstance(pair, list) or len(pair) != 2
            or not isinstance(pair[0], str) or not pair[0].strip()
            or type(pair[1]) is not int or pair[1] < 0 for pair in held):
            raise ValueError("Campaign group has an invalid held-out split")
        if len({tuple(pair) for pair in held}) != len(held):
            raise ValueError("Campaign has duplicate held-out instances")
        current = sorted(held)
        if normalized is None: normalized = current
        elif normalized != current: raise ValueError("Campaign groups disagree on held-out instances")
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episodes-file", type=Path)
    selection.add_argument("--episode-audit-report", type=Path,
                           help="Original check_subtask_data report; recovered=dropped, replay=selected")
    parser.add_argument("--held-out-instances", type=Path, required=True,
                        help="JSON split list or a campaign whose groups all freeze the same split")
    parser.add_argument("--source-kind", choices=("recovered", "replay", "collected"), required=True)
    parser.add_argument("--root-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--fill-leading-with-first-subtask", action="store_true",
                        help="Backfill only leading empty expert labels in the derived training index; preserve source labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("Recovery index output already exists")
    audit = None
    if args.episode_audit_report:
        if args.source_kind == "collected":
            parser.error("Collected sources require explicit --episodes-file; an audit may include original replay")
        audit = json.loads(args.episode_audit_report.read_text())
        key = "dropped_episode_indices" if args.source_kind == "recovered" else "selected_episode_indices"
        if not isinstance(audit, dict) or not isinstance(audit.get(key), list) or not audit[key]:
            raise ValueError(f"Episode audit report lacks nonempty {key}")
        if audit.get("status", "passed") != "passed":
            raise ValueError("Original episode audit did not pass")
        expected_label_sha = audit.get("training_contract", {}).get("annotation_sha256")
        if expected_label_sha is not None and expected_label_sha != file_sha256(args.labels):
            raise ValueError("Original audit annotation identity differs from supplied labels")
        ids = audit[key]
    else:
        try:
            ids = [int(line) for line in args.episodes_file.read_text().splitlines()]
        except ValueError as error:
            raise ValueError("Episode file must contain one integer ID per line") from error
    index = prepare_index(args.dataset_root, args.labels, ids, repo_id=args.repo_id, root_id=args.root_id,
        source_kind=args.source_kind, held_out_instances=held_out_from_file(args.held_out_instances),
        action_horizon=args.action_horizon,
        fill_leading_with_first_subtask=args.fill_leading_with_first_subtask)
    index["provenance"]["selection_input"] = dict(
        path=str((args.episode_audit_report or args.episodes_file).resolve()),
        sha256=file_sha256(args.episode_audit_report or args.episodes_file),
        kind="episode_audit_report" if audit is not None else "explicit_episode_file")
    index["provenance"]["held_out_input"] = dict(path=str(args.held_out_instances.resolve()),
                                                 sha256=file_sha256(args.held_out_instances))
    index["fingerprint"] = canonical_digest({key: value for key, value in index.items() if key != "fingerprint"})
    verify_source_files(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(fingerprint=index["fingerprint"], source_kind=index["source_kind"],
                         **index["provenance"]["statistics"])))


if __name__ == "__main__": main()
