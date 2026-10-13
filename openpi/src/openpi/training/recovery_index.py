"""Metadata-only, immutable indices into original labelled B1K intervals.

Intervals retain original episode-local [start, end) label boundaries. A legal
anchor satisfies start <= frame <= end - action_horizon; no frame table or
video is materialized while preparing or validating this index.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re


KIND = "b1k_valid_intervals"
SOURCE_KINDS = {"recovered", "replay", "collected"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def canonical_digest(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _holdout(groups) -> list[list]:
    if not isinstance(groups, (list, tuple)):
        raise ValueError("held_out_instances must be an explicit list")
    normalized = []
    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            raise ValueError("Invalid held-out task/instance group")
        normalized.append((_text(group[0], "held-out task"), _integer(group[1], "held-out instance")))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate held-out instance group")
    return [list(group) for group in sorted(normalized)]


def valid_anchor_count(interval: dict, action_horizon: int) -> int:
    return max(0, interval["end"] - interval["start"] - action_horizon + 1)


def _annotated(row: dict) -> bool:
    flags = [row[name] for name in ("annotated", "is_annotated") if name in row]
    if any(type(flag) is not bool for flag in flags) or (len(flags) == 2 and flags[0] != flags[1]):
        raise ValueError("Conflicting or invalid annotation flags")
    annotated = flags[0] if flags else True
    if "provenance" in row:
        provenance = row["provenance"]
        if provenance not in {"annotated", "imputed_leading", "imputed_gap", "imputed_tail"}:
            raise ValueError("Invalid annotation provenance")
        if annotated != (provenance == "annotated"):
            raise ValueError("Annotation flag differs from provenance")
    return annotated


def build_index(metadata: Sequence[dict], segments: Sequence[dict], *, dataset_root: str | Path,
                repo_id: str, root_id: str, source_kind: str, episode_indices: Sequence[int],
                held_out_instances=(), action_horizon: int = 32, provenance: dict | None = None,
                fill_leading_with_first_subtask: bool = False) -> dict:
    """Validate complete source tiling, then index trusted nonempty intervals."""
    if source_kind not in SOURCE_KINDS:
        raise ValueError("Invalid recovery source kind")
    if type(fill_leading_with_first_subtask) is not bool:
        raise ValueError("Leading-label filling must be an explicit boolean")
    if fill_leading_with_first_subtask and source_kind != "recovered":
        raise ValueError("Leading-label filling is only supported for recovered expert data")
    _text(repo_id, "repo_id"); _text(root_id, "root_id")
    _integer(action_horizon, "action_horizon", 1)
    if not Path(dataset_root).is_absolute():
        raise ValueError("dataset_root must be absolute")
    ids = list(episode_indices)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Episode selection must be explicit, nonempty and unique")
    for episode in ids: _integer(episode, "selected episode")
    selected = set(ids)
    holdout = _holdout(held_out_instances)
    reserved = {tuple(group) for group in holdout}
    records = {}
    for row in metadata:
        if not isinstance(row, dict) or not {"episode_index", "length", "task_name", "task_instance_id"} <= row.keys():
            raise ValueError("Missing required episode identity fields")
        ep = _integer(row["episode_index"], "episode_index")
        if ep in records:
            raise ValueError("Duplicate metadata episode identity")
        records[ep] = dict(episode_index=ep, length=_integer(row["length"], "length", 1),
            task_name=_text(row["task_name"], "task_name"),
            task_instance_id=_integer(row["task_instance_id"], "task_instance_id"))
    if selected - records.keys():
        raise ValueError("Unknown selected episode")
    grouped = {ep: [] for ep in selected}
    for row in segments:
        if not isinstance(row, dict) or not {"episode_index", "start", "end", "subtask"} <= row.keys():
            raise ValueError("Missing required source interval fields")
        ep = _integer(row["episode_index"], "segment episode")
        if ep not in selected:
            continue
        start, end = _integer(row["start"], "segment start"), _integer(row["end"], "segment end", 1)
        label = row["subtask"]
        if label is not None and not isinstance(label, str):
            raise ValueError("Subtask label must be a string or null")
        if end <= start or end > records[ep]["length"]:
            raise ValueError("Reversed or out-of-bounds source interval")
        annotated = _annotated(row)
        grouped[ep].append((start, end, None if label is None else label.strip(), annotated))
    evidence = dict(provenance or {})
    collected = {row["episode_index"]: row for row in evidence.get("collected_episodes", [])}
    if len(collected) != len(evidence.get("collected_episodes", [])):
        raise ValueError("Duplicate collected provenance identity")
    episodes, intervals = [], []
    stats = dict(requested_episodes=len(selected), held_out_excluded_episodes=0, trusted_labelled_frames=0,
                 empty_label_frames=0, unannotated_frames=0, short_segments=0, short_segment_frames=0,
                 no_legal_anchor_episodes=0, eligible_episodes=0, eligible_segments=0, valid_anchors=0)
    filled = []
    for ep in sorted(selected):
        rows = sorted(grouped[ep])
        cursor = 0
        for start, end, _, _ in rows:
            if start != cursor:
                raise ValueError(f"Source intervals must exactly tile episode {ep}, including empty labels")
            cursor = end
        if cursor != records[ep]["length"]:
            raise ValueError(f"Missing or incomplete source label coverage for episode {ep}")
        has_empty = any(not label for _, _, label, _ in rows)
        if source_kind == "recovered" and not has_empty:
            raise ValueError("Recovered episode was not dropped for an empty label")
        if source_kind in {"replay", "collected"} and has_empty:
            raise ValueError("Replay/collected sources require nonempty labels throughout each episode")
        if (records[ep]["task_name"], records[ep]["task_instance_id"]) in reserved:
            stats["held_out_excluded_episodes"] += 1
            continue
        filled_end = 0
        if fill_leading_with_first_subtask:
            first = next((i for i, (_, _, label, _) in enumerate(rows) if label), None)
            if first is None or any(not label for _, _, label, _ in rows[first:]):
                raise ValueError("Leading-label filling requires a labelled suffix without empty gaps or tails")
            start, end, label, annotated = rows[first]
            if not annotated:
                raise ValueError("Leading-label filling requires a trusted first nonempty subtask")
            filled_end = start
            # Extend only the first label's interval in this derived index. The
            # source parquet and original human-annotation boundary stay intact.
            rows = [(0, end, label, annotated), *rows[first + 1:]]
            filled.append(dict(episode_index=ep, start=0, end=start, subtask=label,
                               source_label_start=start, source_label_end=end,
                               is_human_annotated=False, provenance="imputed_leading"))
        eligible = []
        for start, end, label, annotated in rows:
            length = end - start
            if not label:
                stats["empty_label_frames"] += length
            elif not annotated:
                stats["unannotated_frames"] += length
            else:
                stats["trusted_labelled_frames"] += length - (filled_end if start == 0 else 0)
                if length < action_horizon:
                    stats["short_segments"] += 1; stats["short_segment_frames"] += length
                else:
                    interval = dict(episode_index=ep, start=start, end=end, subtask=label)
                    if filled_end and start == 0:
                        interval.update(imputed_leading_end=filled_end, original_label_start=filled_end)
                    eligible.append(interval)
        if not eligible:
            stats["no_legal_anchor_episodes"] += 1
            continue
        record = records[ep].copy()
        if source_kind == "collected":
            proof = collected.get(ep)
            if not proof or not _HASH.fullmatch(str(proof.get("content_sha256", ""))):
                raise ValueError("Collected episode requires a verified trajectory content identity")
            if proof.get("source_kind") != "collected" or proof.get("length") != record["length"]:
                raise ValueError("Collected provenance differs from source metadata")
            if (proof.get("task_name"), proof.get("task_instance_id")) != (record["task_name"], record["task_instance_id"]):
                raise ValueError("Collected task/instance provenance differs from metadata")
            record.update(content_sha256=proof["content_sha256"], trial_id=_text(proof.get("trial_id"), "trial_id"))
            if "source_episode_index" in proof:
                record["source_episode_index"] = _integer(proof["source_episode_index"], "source_episode_index")
        episodes.append(record); intervals.extend(eligible)
    if not intervals:
        raise ValueError("No legal labelled anchors remain after source and holdout filtering")
    stats.update(eligible_episodes=len(episodes), eligible_segments=len(intervals),
                 valid_anchors=sum(valid_anchor_count(row, action_horizon) for row in intervals))
    evidence["statistics"] = stats
    if fill_leading_with_first_subtask:
        stats.update(imputed_leading_episodes=len(filled),
                     imputed_leading_frames=sum(row["end"] for row in filled))
        evidence.update(leading_fill_policy="first_nonempty_subtask", leading_fill_records=filled)
    evidence.update(metadata_only=True, videos_read=False, source_labels_modified=False,
                    action_boundary_policy="full_horizon_within_label_interval")
    payload = dict(schema_version=1, kind=KIND, source_kind=source_kind, root_id=root_id, repo_id=repo_id,
        dataset_root=str(dataset_root), action_horizon=action_horizon, held_out_instances=holdout,
        episodes=episodes, intervals=intervals, provenance=evidence)
    result = {**payload, "fingerprint": canonical_digest(payload)}
    validate_index(result)
    return result


def validate_index(index: dict) -> dict:
    """Validate the frozen index without reading any external source files."""
    if index.get("schema_version") != 1 or index.get("kind") != KIND or index.get("source_kind") not in SOURCE_KINDS:
        raise ValueError("Unknown valid-interval index schema")
    if index.get("fingerprint") != canonical_digest({k: v for k, v in index.items() if k != "fingerprint"}):
        raise ValueError("Recovery index fingerprint differs")
    if not {"root_id", "repo_id", "dataset_root", "action_horizon", "held_out_instances", "episodes", "intervals", "provenance"} <= index.keys():
        raise ValueError("Incomplete valid-interval index")
    _text(index["root_id"], "root_id"); _text(index["repo_id"], "repo_id")
    if not Path(index["dataset_root"]).is_absolute(): raise ValueError("Index source root must be absolute")
    horizon = _integer(index["action_horizon"], "action_horizon", 1)
    reserved = {tuple(group) for group in _holdout(index["held_out_instances"])}
    records = {}
    for row in index["episodes"]:
        if not isinstance(row, dict) or not {"episode_index", "length", "task_name", "task_instance_id"} <= row.keys():
            raise ValueError("Incomplete indexed episode identity")
        ep = _integer(row["episode_index"], "episode_index")
        if ep in records: raise ValueError("Duplicate index episode")
        _integer(row["length"], "episode length", 1)
        group = (_text(row["task_name"], "task_name"), _integer(row["task_instance_id"], "task_instance_id"))
        if group in reserved: raise ValueError("Index contains a held-out instance")
        if index["source_kind"] == "collected" and not _HASH.fullmatch(str(row.get("content_sha256", ""))):
            raise ValueError("Collected index lacks content identity")
        records[ep] = row
    if not records or not index["intervals"]: raise ValueError("Empty valid-interval index")
    previous, used = {}, set()
    for row in index["intervals"]:
        if not isinstance(row, dict) or not {"episode_index", "start", "end", "subtask"} <= row.keys():
            raise ValueError("Incomplete indexed interval")
        ep = _integer(row["episode_index"], "interval episode")
        start, end = _integer(row["start"], "interval start"), _integer(row["end"], "interval end", 1)
        _text(row["subtask"], "subtask")
        if ep not in records or end > records[ep]["length"] or valid_anchor_count(row, horizon) == 0:
            raise ValueError("Index interval has no full legal action horizon")
        if start < previous.get(ep, 0): raise ValueError("Overlapping or unordered index intervals")
        previous[ep] = end; used.add(ep)
    if used != set(records): raise ValueError("Episode has no indexed anchors")
    if not isinstance(index.get("provenance"), dict): raise ValueError("Missing source provenance")
    return index


def verify_source_files(index: dict, *, dataset_root: str | Path | None = None, include_labels: bool = True) -> None:
    validate_index(index)
    root = Path(dataset_root or index["dataset_root"])
    files = index["provenance"].get("source_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Recovery index lacks source-file hashes")
    roles = set()
    for source in files:
        role = source.get("role")
        if role not in {"metadata", "labels", "collection_evidence"} or not _HASH.fullmatch(str(source.get("sha256", ""))):
            raise ValueError("Invalid indexed source-file identity")
        roles.add(role)
        if role == "labels" and not include_labels: continue
        path = Path(_text(source.get("path"), "source path"))
        if role != "labels" and (path.is_absolute() or ".." in path.parts):
            raise ValueError("Metadata/evidence source must stay within dataset root")
        path = path if path.is_absolute() else root / path
        if not path.is_file() or file_sha256(path) != source["sha256"]:
            raise ValueError(f"Indexed {role} source file changed or is missing: {path}")
    if not {"metadata", "labels"} <= roles:
        raise ValueError("Index must hash metadata and original labels")


def load_index(path: str | Path, *, verify_sources: bool = True) -> dict:
    index = validate_index(json.loads(Path(path).read_text()))
    if verify_sources: verify_source_files(index)
    return index


def prepare_index(dataset_root: str | Path, labels_path: str | Path, episode_indices: Sequence[int], *,
                  repo_id: str, root_id: str, source_kind: str, held_out_instances=(), action_horizon: int = 32,
                  fill_leading_with_first_subtask: bool = False) -> dict:
    """Read and hash small metadata/label inputs; never open raw parquet or videos."""
    import pyarrow.parquet as pq
    root, labels = Path(dataset_root).resolve(), Path(labels_path).resolve()
    tasks_path, info_path = root / "meta/tasks.jsonl", root / "meta/info.json"
    if not tasks_path.is_file() or not info_path.is_file(): raise ValueError("Missing source task/info metadata")
    tasks = {}
    for line in tasks_path.read_text().splitlines():
        row = json.loads(line); task = _integer(row["task_index"], "task_index")
        if task in tasks: raise ValueError("Duplicate canonical task identity")
        tasks[task] = _text(row["task_name"], "task_name")
    paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths: raise ValueError("Missing LeRobot episode metadata")
    records = []
    for path in paths:
        required = {"episode_index", "length", "task_index", "task_instance_id"}
        if not required <= set(pq.read_schema(path).names): raise ValueError("Missing required episode identity fields")
        for row in pq.read_table(path, columns=sorted(required)).to_pylist():
            task = _integer(row["task_index"], "task_index")
            if task not in tasks: raise ValueError("Unknown canonical task index")
            records.append({**row, "task_name": tasks[task]})
    required = {"episode_index", "start", "end", "subtask"}
    names = set(pq.read_schema(labels).names)
    if not required <= names: raise ValueError("Missing required source label columns")
    cols = sorted(required | (names & {"annotated", "is_annotated", "provenance"}))
    segments = pq.read_table(labels, columns=cols).to_pylist()
    optional_metadata = [p for p in (root / "meta/tasks.parquet", root / "rgb_view_manifest.json") if p.is_file()]
    source_files = [dict(path=p.relative_to(root).as_posix(), sha256=file_sha256(p), role="metadata")
                    for p in [info_path, tasks_path, *paths, *optional_metadata]]
    source_files.append(dict(path=str(labels), sha256=file_sha256(labels), role="labels"))
    provenance = dict(source_files=source_files, labels_path=str(labels), label_sha256=file_sha256(labels),
                      episode_selection_sha256=canonical_digest({"episode_indices": sorted(episode_indices)}))
    if source_kind == "collected":
        proof_path = root / "meta/collection_provenance.json"
        if not proof_path.is_file(): raise ValueError("Collected source lacks trajectory provenance")
        evidence = [proof_path, root / "_EXPORTED", root / "_VERIFIED_EXPORT", root / "format_verification.json", root / "export_report.json"]
        if any(not path.is_file() for path in evidence): raise ValueError("Collected source lacks verified export evidence")
        report = json.loads((root / "export_report.json").read_text())
        verification = json.loads((root / "format_verification.json").read_text())
        plan_hash = report.get("plan_fingerprint")
        if not _HASH.fullmatch(str(plan_hash or "")) or verification.get("complete") is not True or verification.get("real_lerobot_reader") is not True:
            raise ValueError("Collected format evidence is incomplete")
        if verification.get("kind") != "collection_export_format_verification" or verification.get("plan_fingerprint") != plan_hash:
            raise ValueError("Collected format evidence has a different source identity")
        for marker_path in (root / "_EXPORTED", root / "_VERIFIED_EXPORT"):
            marker = json.loads(marker_path.read_text())
            if marker.get("plan_fingerprint") != plan_hash or marker.get("label_sha256") != report.get("label_sha256"):
                raise ValueError("Collected export marker differs from frozen provenance")
        actual_label_digest = hashlib.sha256(json.dumps(pq.read_table(labels).to_pylist(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if report.get("label_sha256") != actual_label_digest or verification.get("aligned_source_frames_verified") != report.get("new_frames"):
            raise ValueError("Collected labels or source-frame verification differ")
        provenance["collected_episodes"] = json.loads(proof_path.read_text())
        provenance["source_files"].extend(dict(path=p.relative_to(root).as_posix(), sha256=file_sha256(p),
                                              role="collection_evidence") for p in evidence)
    return build_index(records, segments, dataset_root=root, repo_id=repo_id, root_id=root_id,
        source_kind=source_kind, episode_indices=episode_indices, held_out_instances=held_out_instances,
        action_horizon=action_horizon, provenance=provenance,
        fill_leading_with_first_subtask=fill_leading_with_first_subtask)
