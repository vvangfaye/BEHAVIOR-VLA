"""Deterministic, interval-based sampling for staged subtask fine-tuning.

The dataset index is a *global sample index*, before rank partitioning.  Never
shuffle it with a DistributedSampler: the plan already randomizes every choice.
Intervals describe labelled segments [start, end); a horizon H permits anchors
start <= frame <= end - H.  No frame-sized arrays are constructed.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE_KINDS = frozenset({"recovered", "replay", "collected", "full_episode"})
SOURCE_IDS = {"replay": 0, "recovered": 1, "collected": 2, "full_episode": 3}
CONTRACT_FILENAME = "subtask_mixture.json"
SAMPLER_VERSION = 1


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _held_out(value: Any) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise ValueError("held_out_instances must be a list")
    result = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("Invalid held-out task-instance pair")
        result.append((_text(row[0], "held-out task"), _integer(row[1], "held-out instance")))
    if len(set(result)) != len(result):
        raise ValueError("Duplicate held-out task-instance pair")
    return tuple(sorted(result))


def _index_identity(index: dict) -> str:
    # Relocation and JSON formatting are transport changes, not sample changes.
    semantic = {k: v for k, v in index.items() if k not in {"fingerprint", "dataset_root"}}
    return _digest(semantic)


def _validate_index(index: dict, horizon: int) -> None:
    if not isinstance(index, dict) or index.get("schema_version") != 1 or index.get("kind") != "b1k_valid_intervals":
        raise ValueError("Unsupported valid-interval index")
    fingerprint = _sha(index.get("fingerprint"), "index fingerprint")
    if fingerprint != _digest({k: v for k, v in index.items() if k != "fingerprint"}):
        raise ValueError("Index fingerprint mismatch")
    for key in ("root_id", "repo_id", "dataset_root"):
        _text(index.get(key), key)
    if index.get("source_kind") not in SOURCE_KINDS:
        raise ValueError("Unknown index source_kind")
    weak_text = index.get("provenance", {}).get("label_semantics") == "self_generated_weak_subtask"
    if index["source_kind"] == "full_episode":
        if not weak_text or index.get("provenance", {}).get("language_loss_enabled") is not False:
            raise ValueError("Full episodes require explicit weak text and disabled language supervision")
    elif weak_text:
        raise ValueError("Weak full-episode text cannot enter a supervised source")
    if _integer(index.get("action_horizon"), "index action_horizon", minimum=1) != horizon:
        raise ValueError("Index action_horizon differs from mixture")
    held = set(_held_out(index.get("held_out_instances")))
    if not isinstance(index.get("provenance"), dict):
        raise ValueError("Index provenance must be an object")
    episodes = index.get("episodes")
    intervals = index.get("intervals")
    if not isinstance(episodes, list) or not episodes or not isinstance(intervals, list) or not intervals:
        raise ValueError("Every index needs episodes and valid intervals")
    records = {}
    for episode in episodes:
        number = _integer(episode.get("episode_index"), "episode_index")
        if number in records:
            raise ValueError("Duplicate episode in index")
        _integer(episode.get("length"), "episode length", minimum=1)
        task = _text(episode.get("task_name"), "task_name")
        instance = _integer(episode.get("task_instance_id"), "task_instance_id")
        if (task, instance) in held:
            raise ValueError("Training index contains a held-out instance")
        if index["source_kind"] in {"collected", "full_episode"}:
            _sha(episode.get("content_sha256"), "collected episode content_sha256")
        records[number] = episode
    seen = set()
    spans: dict[int, list[tuple[int, int]]] = {}
    for interval in intervals:
        number = _integer(interval.get("episode_index"), "interval episode_index")
        if number not in records:
            raise ValueError("Interval refers to an absent episode")
        start = _integer(interval.get("start"), "interval start")
        end = _integer(interval.get("end"), "interval end", minimum=1)
        _text(interval.get("subtask"), "interval subtask")
        if end > records[number]["length"] or end - start < horizon:
            raise ValueError("Interval has no complete action window or exceeds episode")
        spans.setdefault(number, []).append((start, end))
        seen.add(number)
    if seen != set(records):
        raise ValueError("Index contains episodes without valid intervals")
    for values in spans.values():
        values.sort()
        if any(a[1] > b[0] for a, b in zip(values, values[1:])):
            raise ValueError("Overlapping or duplicate labelled intervals")


class MixturePlan:
    """Load, validate and sample a finite three-source curriculum.

    A phase uses ``sources=[{kind, weight, indexes:[{path, sha256}]}]``.
    A kind appears at most once per phase; all its indexes share its weight.
    Optional source ``group_weights=[{task_name, subtask, weight}]`` increases
    specified groups' probability; unspecified groups retain weight one.
    ``roots`` and ``episodes_by_root`` allow a loader to open each physical root
    once. ``reference`` returns an episode-local frame, not a flattened offset.
    """

    @classmethod
    def load(cls, path: str | Path) -> "MixturePlan":
        path = Path(path)
        manifest = json.loads(path.read_text())
        return cls(manifest, base_dir=path.resolve().parent)

    def __init__(self, manifest: dict, *, base_dir: Path):
        self.manifest = copy.deepcopy(manifest)
        self.four_source = (manifest.get("schema_version"), manifest.get("kind")) == (2, "b1k_four_source_mixture")
        if not self.four_source and (manifest.get("schema_version"), manifest.get("kind")) != (1, "b1k_three_source_mixture"):
            raise ValueError("Unsupported mixture manifest")
        self.sampler_version = 2 if self.four_source else SAMPLER_VERSION
        self.seed = _integer(manifest.get("seed"), "seed")
        self.global_batch_size = _integer(manifest.get("global_batch_size"), "global_batch_size", minimum=1)
        self.action_horizon = _integer(manifest.get("action_horizon"), "action_horizon", minimum=1)
        self.roots: dict[str, dict] = {}
        self.episodes_by_root: dict[str, list[int]] = {}
        self.episode_records_by_root: dict[str, dict[int, dict]] = {}
        self.indexes: list[dict] = []
        self._phases: list[dict] = []
        self._ends: list[int] = []
        self._identity_indexes: dict[str, dict] = {}
        self._file_indexes: dict[tuple[str, str], dict] = {}
        self._registered_episodes: dict[tuple[str, int], str] = {}
        self._original_episodes: dict[tuple[str, int, int], str] = {}
        self._collected_content: dict[str, tuple[str, int]] = {}
        self._held_out: tuple[tuple[str, int], ...] | None = None
        phases = manifest.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("Mixture requires nonempty phases")
        expected_start = 0
        for phase in phases:
            start = _integer(phase.get("start_step"), "start_step")
            end = _integer(phase.get("end_step"), "end_step", minimum=1)
            if start != expected_start or end <= start:
                raise ValueError("Phases must be contiguous, ordered and start at step zero")
            expected_start = end
            sources = phase.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError("Phase requires sources")
            kinds = set()
            compiled_sources = []
            for source in sources:
                kind = source.get("kind")
                if kind not in SOURCE_KINDS or kind in kinds or (kind == "full_episode" and not self.four_source):
                    raise ValueError("Unknown or repeated source kind; place multiple roots in indexes")
                kinds.add(kind)
                weight = source.get("weight")
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                    raise ValueError("Source weight must be finite and nonnegative")
                refs = source.get("indexes")
                if not isinstance(refs, list) or not refs:
                    raise ValueError("Source requires nonempty indexes")
                indexes = []
                identities = set()
                for ref in refs:
                    relative = _text(ref.get("path"), "index path")
                    expected_sha = _sha(ref.get("sha256"), "index sha256")
                    index_path = (base_dir / relative).resolve()
                    file_key = (str(index_path), expected_sha)
                    if file_key not in self._file_indexes:
                        raw = index_path.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != expected_sha:
                            raise ValueError("Index file SHA256 mismatch")
                        index = json.loads(raw)
                        _validate_index(index, self.action_horizon)
                        self._file_indexes[file_key] = index
                    index = self._file_indexes[file_key]
                    if index["source_kind"] != kind:
                        raise ValueError("Manifest source kind does not match index")
                    identity = _index_identity(index)
                    if identity in identities:
                        raise ValueError("Repeated index in one source")
                    identities.add(identity)
                    self._register(index, identity)
                    indexes.append(index)
                groups: dict[tuple[str, str], dict[int, list[dict]]] = {}
                for index in indexes:
                    records = {e["episode_index"]: e for e in index["episodes"]}
                    for interval in index["intervals"]:
                        episode = records[interval["episode_index"]]
                        group = (episode["task_name"], interval["subtask"])
                        instance = episode["task_instance_id"]
                        segment = {**interval, "root_id": index["root_id"], "task_name": episode["task_name"],
                                   "task_instance_id": instance}
                        groups.setdefault(group, {}).setdefault(instance, []).append(segment)
                compiled_groups = []
                for group, instances in sorted(groups.items()):
                    compiled_instances = []
                    for instance, segments in sorted(instances.items()):
                        segments.sort(key=lambda s: (s["root_id"], s["episode_index"], s["start"], s["end"]))
                        compiled_instances.append((instance, tuple(segments)))
                    compiled_groups.append((group, tuple(compiled_instances)))
                group_overrides = source.get("group_weights", [])
                if not isinstance(group_overrides, list):
                    raise ValueError("group_weights must be a list")
                weights_by_group = {}
                for override in group_overrides:
                    group = (_text(override.get("task_name"), "group task_name"),
                             _text(override.get("subtask"), "group subtask"))
                    group_weight = override.get("weight")
                    if (isinstance(group_weight, bool) or not isinstance(group_weight, (int, float))
                            or not math.isfinite(group_weight) or group_weight <= 0):
                        raise ValueError("Group weight must be finite and positive")
                    if group not in groups or group in weights_by_group:
                        raise ValueError("Unknown or repeated weighted task/subtask group")
                    weights_by_group[group] = float(group_weight)
                sampling = source.get("group_sampling", "uniform_groups")
                if sampling not in {"uniform_groups", "task_balanced"}:
                    raise ValueError("Unknown group_sampling")
                per_task = {}
                for group, _ in compiled_groups:
                    per_task[group[0]] = per_task.get(group[0], 0) + 1
                group_weights = [weights_by_group.get(group, 1.0) /
                                 (per_task[group[0]] if sampling == "task_balanced" else 1)
                                 for group, _ in compiled_groups]
                group_total = math.fsum(group_weights)
                if not math.isfinite(group_total):
                    raise ValueError("Total group weight must be finite")
                cumulative, group_cdf = 0.0, []
                for value in group_weights:
                    cumulative += value / group_total
                    group_cdf.append(cumulative)
                compiled_sources.append({"kind": kind, "weight": float(weight), "indexes": sorted(identities),
                                         "groups": tuple(compiled_groups), "group_cdf": tuple(group_cdf),
                                         "uniform_groups": len(set(group_weights)) == 1,
                                         "group_weights": [
                                             {"task_name": group[0], "subtask": group[1], "weight": value / group_total}
                                             for (group, _), value in zip(compiled_groups, group_weights)]})
            if self.four_source and kinds != SOURCE_KINDS:
                raise ValueError("Four-source phases must declare all four source kinds")
            compiled_sources.sort(key=lambda s: s["kind"])
            total = math.fsum(s["weight"] for s in compiled_sources)
            if not total > 0 or not math.isfinite(total):
                raise ValueError("Phase needs positive total source weight")
            for source in compiled_sources:
                source["weight"] /= total
            self._phases.append({"start_step": start, "end_step": end, "sources": compiled_sources})
            self._ends.append(end * self.global_batch_size)
        self.num_steps = expected_start
        self.num_samples = self.num_steps * self.global_batch_size
        self.episodes_by_root = {root: sorted(records) for root, records in self.episode_records_by_root.items()}
        self.fingerprint = _digest(self._semantic_plan())

    def _register(self, index: dict, identity: str) -> None:
        held = _held_out(index["held_out_instances"])
        if self._held_out is not None and held != self._held_out:
            raise ValueError("Indexes disagree on held-out instances")
        self._held_out = held
        root = index["root_id"]
        root_record = {"repo_id": index["repo_id"], "dataset_root": index["dataset_root"]}
        if root in self.roots and self.roots[root] != root_record:
            raise ValueError("One root_id has conflicting repository or dataset location")
        self.roots[root] = root_record
        if identity in self._identity_indexes:
            return  # The same frozen index is deliberately reused in later phases.
        self._identity_indexes[identity] = index
        self.indexes.append(copy.deepcopy(index))
        records = self.episode_records_by_root.setdefault(root, {})
        for episode in index["episodes"]:
            number = episode["episode_index"]
            key = (root, number)
            if key in self._registered_episodes:
                raise ValueError("Overlapping or conflicting episode indexes within one root")
            self._registered_episodes[key] = identity
            records[number] = copy.deepcopy(episode)
            kind = index["source_kind"]
            if kind in {"collected", "full_episode"}:
                content = episode["content_sha256"]
                if content in self._collected_content:
                    raise ValueError("Duplicate collected trajectory content across indexes")
                self._collected_content[content] = key
            else:
                original = _integer(episode.get("source_episode_index", number), "source_episode_index")
                source_key = (episode["task_name"], episode["task_instance_id"], original)
                if source_key in self._original_episodes:
                    raise ValueError("Original episode overlaps recovered/replay indexes or roots")
                self._original_episodes[source_key] = kind

    def reference(self, global_index: int) -> dict:
        """Return one reproducible sample, independent of rank or iteration order."""
        global_index = _integer(global_index, "global_index")
        if global_index >= self.num_samples:
            raise IndexError("Mixture global sample index is out of range")
        phase = self._phases[bisect.bisect_right(self._ends, global_index)]

        def draw(label: str) -> int:
            payload = f"b1k-mixture-v{self.sampler_version}:{self.seed}:{global_index}:{label}".encode()
            return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

        quantile = draw("source") / 2**64
        cumulative = 0.0
        active = [s for s in phase["sources"] if s["weight"] > 0]
        source = active[-1]
        for candidate in active:
            cumulative += candidate["weight"]
            if quantile < cumulative:
                source = candidate
                break
        groups = source["groups"]
        if source["uniform_groups"]:
            group_index = draw("group") % len(groups)
        else:
            group_index = min(bisect.bisect_right(source["group_cdf"], draw("group") / 2**64), len(groups) - 1)
        group, instances = groups[group_index]
        _, segments = instances[draw("instance") % len(instances)]
        segment = segments[draw("segment") % len(segments)]
        anchors = segment["end"] - segment["start"] - self.action_horizon + 1
        frame = segment["start"] + draw("anchor") % anchors
        return {"root_id": segment["root_id"], "episode_index": segment["episode_index"], "frame": frame,
                "subtask": segment["subtask"], "source_kind": source["kind"], "task_name": segment["task_name"],
                "task_instance_id": segment["task_instance_id"], "group": group,
                "segment_start": segment["start"], "segment_end": segment["end"], "global_index": global_index}

    def _semantic_plan(self) -> dict:
        return {"sampler_version": self.sampler_version, "seed": self.seed, "global_batch_size": self.global_batch_size,
                "action_horizon": self.action_horizon, "phases": [
                    {"start_step": p["start_step"], "end_step": p["end_step"], "sources": [
                        {"kind": s["kind"], "weight": s["weight"], "indexes": s["indexes"],
                         "group_weights": s["group_weights"]} for s in p["sources"]]}
                    for p in self._phases]}


def _prefix(semantic_plan: dict, step: int) -> dict:
    step = _integer(step, "consumed step")
    if step > semantic_plan["phases"][-1]["end_step"]:
        raise ValueError("Consumed step exceeds the finite mixture plan")
    result = {k: copy.deepcopy(v) for k, v in semantic_plan.items() if k != "phases"}
    result["phases"] = []
    for phase in semantic_plan["phases"]:
        if phase["start_step"] >= step:
            break
        clipped = copy.deepcopy(phase)
        clipped["end_step"] = min(step, phase["end_step"])
        previous = result["phases"][-1] if result["phases"] else None
        if previous is not None and previous["sources"] == clipped["sources"]:
            previous["end_step"] = clipped["end_step"]
        else:
            result["phases"].append(clipped)
    return result


def consumed_contract(plan: MixturePlan, step: int) -> dict:
    """Checkpoint at completed optimizer step ``step`` (not directory step)."""
    semantic = plan._semantic_plan()
    prefix = _prefix(semantic, step)
    result = {"schema_version": 1, "kind": "b1k_consumed_mixture_contract", "step": step,
              "consumed_samples": step * plan.global_batch_size, "manifest": copy.deepcopy(plan.manifest),
              "indexes": copy.deepcopy(plan.indexes), "semantic_plan": semantic,
              "consumed_prefix_sha256": _digest(prefix)}
    result["fingerprint"] = _digest(result)
    return result


def compare_consumed_contract(saved: dict, plan: MixturePlan, step: int | None = None) -> None:
    """Reject changes to consumed samples; permit only future phase changes.

    If ``step`` is supplied it must be the restored train_state.step.  Index JSON
    formatting/path relocation does not change identity, but source bytes and
    index fingerprints are always independently checked when loading a plan.
    """
    if saved.get("schema_version") != 1 or saved.get("kind") != "b1k_consumed_mixture_contract":
        raise ValueError("Missing or unsupported consumed mixture contract")
    if saved.get("fingerprint") != _digest({k: v for k, v in saved.items() if k != "fingerprint"}):
        raise ValueError("Consumed mixture contract fingerprint mismatch")
    consumed_step = _integer(saved.get("step"), "saved consumed step")
    if step is not None and _integer(step, "restored step") != consumed_step:
        raise ValueError("Mixture contract step differs from restored optimizer step")
    semantic = saved["semantic_plan"]
    if saved.get("consumed_samples") != consumed_step * semantic["global_batch_size"]:
        raise ValueError("Consumed sample count differs from optimizer step")
    saved_prefix = _prefix(semantic, consumed_step)
    if saved.get("consumed_prefix_sha256") != _digest(saved_prefix):
        raise ValueError("Saved mixture consumed prefix is inconsistent")
    if _prefix(plan._semantic_plan(), consumed_step) != saved_prefix:
        raise ValueError("Cannot resume: already-consumed mixture sampling changed")


def save_consumed_contract(directory: str | Path, plan: MixturePlan, step: int) -> dict:
    """Write inside the checkpoint's atomic assets save transaction."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = consumed_contract(plan, step)
    (directory / CONTRACT_FILENAME).write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    return result


def check_consumed_contract(directory: str | Path, plan: MixturePlan, step: int | None = None) -> None:
    path = Path(directory) / CONTRACT_FILENAME
    if not path.is_file():
        raise ValueError("Mixture resume requires a saved consumed sampling contract")
    compare_consumed_contract(json.loads(path.read_text()), plan, step)
