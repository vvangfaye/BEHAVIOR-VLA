#!/usr/bin/env python3
import argparse
import contextlib
import dataclasses
import errno
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

JOB_STATES = ("pending", "running", "succeeded", "exhausted")
RESULT_REQUIRED_KEYS = {"task", "instance_id", "rollout_id", "steps", "success"}
VIDEO_REQUIRED = "required"
VIDEO_REQUIRED_IF_ELIGIBLE = "required_if_eligible"
_LEGACY_VIDEO_OPTIONAL = "legacy_optional"


@dataclasses.dataclass(frozen=True)
class JobLease:
    task: str
    instance_id: int
    attempt: int
    worker: str
    lease_token: str
    attempt_dir: Path

    @property
    def json_dir(self) -> Path:
        return self.attempt_dir / "json"

    @property
    def video_dir(self) -> Path:
        return self.attempt_dir / "videos"

    def result_path(self, rollout_id: int) -> Path:
        return self.json_dir / f"{self.task}_{self.instance_id}_{rollout_id}.json"

    def video_path(self, rollout_id: int) -> Path:
        return self.video_dir / f"{self.task}_{self.instance_id}_{rollout_id}.mp4"


class LeaseLostError(RuntimeError):
    pass


class ArtifactValidationError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, relative_to: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": stat.st_size,
        "sha256": _sha256(path),
    }


class EvaluationQueue:
    def __init__(self, db_path: str | Path, busy_timeout_ms: int = 30_000):
        self.db_path = Path(db_path).expanduser().resolve()
        self.busy_timeout_ms = busy_timeout_ms

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=self.busy_timeout_ms / 1000,
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path, timeout=self.busy_timeout_ms / 1000
            )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        if not readonly:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextlib.contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(
        self,
        jobs: Iterable[tuple[str, int]],
        *,
        max_attempts: int,
        metadata: dict | None = None,
    ) -> None:
        jobs = list(jobs)
        if not jobs:
            raise ValueError("jobs must not be empty")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        normalized = []
        for task, instance_id in jobs:
            if not isinstance(task, str) or not task.strip():
                raise ValueError("job task must be a nonempty string")
            if (
                isinstance(instance_id, bool)
                or not isinstance(instance_id, int)
                or instance_id < 0
            ):
                raise ValueError("job instance_id must be a non-negative integer")
            normalized.append((task.strip(), instance_id))
        if len(normalized) != len(set(normalized)):
            raise ValueError("jobs must contain unique task and instance_id pairs")

        schema = (
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                task TEXT NOT NULL,
                instance_id INTEGER NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'exhausted')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
                worker TEXT,
                lease_token TEXT,
                attempt_dir TEXT,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                succeeded_at REAL,
                provenance TEXT,
                UNIQUE(task, instance_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attempts (
                job_id INTEGER NOT NULL REFERENCES jobs(id),
                attempt INTEGER NOT NULL,
                worker TEXT NOT NULL,
                lease_token TEXT NOT NULL UNIQUE,
                attempt_dir TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('running', 'failed', 'succeeded')),
                started_at REAL NOT NULL,
                ended_at REAL,
                error TEXT,
                PRIMARY KEY(job_id, attempt)
            )
            """,
            "CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(task, state, attempts, instance_id)",
            "CREATE INDEX IF NOT EXISTS jobs_worker_idx ON jobs(worker, state)",
        )
        with self._immediate() as connection:
            for statement in schema:
                connection.execute(statement)
            existing = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if existing:
                raise RuntimeError(f"queue is already initialized with {existing} jobs")
            timestamp = _now()
            connection.executemany(
                """
                INSERT INTO jobs(task, instance_id, state, max_attempts, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?)
                """,
                [
                    (task, instance_id, max_attempts, timestamp, timestamp)
                    for task, instance_id in normalized
                ],
            )
            for key, value in sorted((metadata or {}).items()):
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (str(key), _json_text(value)),
                )

    def metadata(self, *, readonly: bool = True) -> dict:
        with self._connect(readonly=readonly) as connection:
            rows = connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            ).fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def claim(
        self, task: str, worker: str, attempt_root: str | Path
    ) -> JobLease | None:
        if not task.strip() or not worker.strip():
            raise ValueError("task and worker must be nonempty")
        attempt_root = Path(attempt_root).expanduser().resolve()
        with self._immediate() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE task = ? AND state = 'pending' AND attempts < max_attempts
                ORDER BY CASE WHEN attempts = 0 THEN 0 ELSE 1 END, attempts, instance_id
                LIMIT 1
                """,
                (task,),
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempts"]) + 1
            lease_token = uuid.uuid4().hex
            attempt_dir = (
                attempt_root
                / task
                / str(row["instance_id"])
                / f"attempt-{attempt:02d}-{lease_token}"
            )
            timestamp = _now()
            changed = connection.execute(
                """
                UPDATE jobs
                SET state = 'running', attempts = ?, worker = ?, lease_token = ?, attempt_dir = ?,
                    last_error = NULL, updated_at = ?
                WHERE id = ? AND state = 'pending' AND attempts = ?
                """,
                (
                    attempt,
                    worker,
                    lease_token,
                    str(attempt_dir),
                    timestamp,
                    row["id"],
                    row["attempts"],
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("claim lost despite BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO attempts(job_id, attempt, worker, lease_token, attempt_dir, state, started_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (row["id"], attempt, worker, lease_token, str(attempt_dir), timestamp),
            )
        lease = JobLease(
            task=task,
            instance_id=int(row["instance_id"]),
            attempt=attempt,
            worker=worker,
            lease_token=lease_token,
            attempt_dir=attempt_dir,
        )
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
        except BaseException as exc:
            self.fail(lease, f"could not create attempt directory: {exc}")
            raise
        return lease

    @staticmethod
    def _lease_where(lease: JobLease) -> tuple:
        return (
            lease.task,
            lease.instance_id,
            lease.worker,
            lease.lease_token,
            lease.attempt,
        )

    def _assert_lease(
        self, connection: sqlite3.Connection, lease: JobLease
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM jobs
            WHERE task = ? AND instance_id = ? AND state = 'running'
                AND worker = ? AND lease_token = ? AND attempts = ?
            """,
            self._lease_where(lease),
        ).fetchone()
        if row is None:
            raise LeaseLostError(
                f"lease is no longer active for {lease.task}/{lease.instance_id} attempt {lease.attempt}"
            )
        return row

    def fail(self, lease: JobLease, error: str) -> str:
        error = str(error).strip() or "unknown worker failure"
        with self._immediate() as connection:
            row = self._assert_lease(connection, lease)
            state = "pending" if row["attempts"] < row["max_attempts"] else "exhausted"
            timestamp = _now()
            connection.execute(
                """
                UPDATE jobs
                SET state = ?, worker = NULL, lease_token = NULL, attempt_dir = NULL,
                    last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (state, error, timestamp, row["id"]),
            )
            connection.execute(
                """
                UPDATE attempts SET state = 'failed', ended_at = ?, error = ?
                WHERE job_id = ? AND attempt = ? AND lease_token = ? AND state = 'running'
                """,
                (timestamp, error, row["id"], lease.attempt, lease.lease_token),
            )
        return state

    def running_leases(
        self, worker: str | None = None, *, readonly: bool = False
    ) -> list[JobLease]:
        query = "SELECT * FROM jobs WHERE state = 'running'"
        values: tuple = ()
        if worker is not None:
            query += " AND worker = ?"
            values = (worker,)
        query += " ORDER BY task, instance_id"
        with self._connect(readonly=readonly) as connection:
            rows = connection.execute(query, values).fetchall()
        return [
            JobLease(
                task=row["task"],
                instance_id=row["instance_id"],
                attempt=row["attempts"],
                worker=row["worker"],
                lease_token=row["lease_token"],
                attempt_dir=Path(row["attempt_dir"]),
            )
            for row in rows
        ]

    @staticmethod
    def _validate_result(
        value: object, task: str, instance_id: int, rollout_id: int
    ) -> None:
        if not isinstance(value, dict):
            raise ArtifactValidationError("rollout result must be a JSON object")
        missing = RESULT_REQUIRED_KEYS - value.keys()
        if missing:
            raise ArtifactValidationError(
                f"rollout result is missing keys: {sorted(missing)}"
            )
        if (value["task"], value["instance_id"], value["rollout_id"]) != (
            task,
            instance_id,
            rollout_id,
        ):
            raise ArtifactValidationError(
                "rollout result identity does not match "
                f"{(task, instance_id, rollout_id)!r}"
            )
        if not isinstance(value["task"], str) or not value["task"].strip():
            raise ArtifactValidationError("rollout task must be a nonempty string")
        for name in ("instance_id", "rollout_id", "steps"):
            if (
                isinstance(value[name], bool)
                or not isinstance(value[name], int)
                or value[name] < 0
            ):
                raise ArtifactValidationError(
                    f"rollout {name} must be a non-negative integer"
                )
        if not isinstance(value["success"], bool):
            raise ArtifactValidationError("rollout success must be a boolean")
        q_score = value.get("q_score", 0.0)
        if isinstance(q_score, dict):
            q_score = q_score.get("final", 0.0)
        try:
            q_score = float(q_score or 0.0)
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError("rollout q_score must be numeric") from exc
        if not math.isfinite(q_score):
            raise ArtifactValidationError("rollout q_score must be finite")

    @staticmethod
    def _normalize_rollout_specs(
        *,
        num_rollouts: int | None,
        require_videos: bool | None,
        rollout_specs: Sequence[Mapping[str, object]] | None,
    ) -> list[dict[str, object]]:
        """Normalize the explicit spec, or synthesize one from the legacy arguments.

        Explicit specs use ``video="required"`` or
        ``video="required_if_eligible"``. The latter only permits a missing video
        when the result has ``metric_eligible=false`` or
        ``result_state="setup_excluded"``.
        """
        if rollout_specs is None:
            if (
                isinstance(num_rollouts, bool)
                or not isinstance(num_rollouts, int)
                or num_rollouts <= 0
            ):
                raise ValueError("num_rollouts must be positive")
            if not isinstance(require_videos, bool):
                raise ValueError(
                    "require_videos must be a boolean when rollout_specs is omitted"
                )
            video = VIDEO_REQUIRED if require_videos else _LEGACY_VIDEO_OPTIONAL
            return [
                {"rollout_id": rollout_id, "video": video}
                for rollout_id in range(num_rollouts)
            ]

        if isinstance(rollout_specs, (str, bytes)) or not isinstance(
            rollout_specs, Sequence
        ):
            raise ValueError("rollout_specs must be an ordered sequence of mappings")
        if not rollout_specs:
            raise ValueError("rollout_specs must not be empty")
        if num_rollouts is not None:
            if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, int):
                raise ValueError("num_rollouts must be a positive integer")
            if num_rollouts != len(rollout_specs):
                raise ValueError(
                    "num_rollouts must match the number of rollout_specs when both are supplied"
                )
        if require_videos is not None and not isinstance(require_videos, bool):
            raise ValueError("require_videos must be a boolean")

        normalized = []
        allowed_keys = {"rollout_id", "sample_id", "video"}
        for expected_rollout_id, spec in enumerate(rollout_specs):
            if not isinstance(spec, Mapping):
                raise ValueError("each rollout spec must be a mapping")
            unknown_keys = set(spec) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"rollout spec has unknown keys: {sorted(unknown_keys)}"
                )
            if "rollout_id" not in spec or "video" not in spec:
                raise ValueError("each rollout spec requires rollout_id and video")
            rollout_id = spec["rollout_id"]
            if (
                isinstance(rollout_id, bool)
                or not isinstance(rollout_id, int)
                or rollout_id != expected_rollout_id
            ):
                raise ValueError(
                    "rollout_specs must have ordered dense rollout_id values starting at 0"
                )
            video = spec["video"]
            if video not in (VIDEO_REQUIRED, VIDEO_REQUIRED_IF_ELIGIBLE):
                raise ValueError(
                    "rollout spec video must be 'required' or 'required_if_eligible'"
                )
            normalized_spec = {"rollout_id": rollout_id, "video": video}
            if "sample_id" in spec:
                sample_id = spec["sample_id"]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("rollout spec sample_id must be a nonempty string")
                normalized_spec["sample_id"] = sample_id
            normalized.append(normalized_spec)
        return normalized

    @staticmethod
    def _result_is_metric_eligible(value: dict) -> bool:
        metric_eligible = value.get("metric_eligible")
        result_state = value.get("result_state")
        if not isinstance(metric_eligible, bool):
            raise ArtifactValidationError("rollout metric_eligible must be a boolean")
        if result_state not in ("evaluated", "setup_excluded"):
            raise ArtifactValidationError(
                "rollout result_state must be 'evaluated' or 'setup_excluded'"
            )
        if metric_eligible is not (result_state == "evaluated"):
            raise ArtifactValidationError(
                "rollout metric_eligible conflicts with result_state"
            )
        return metric_eligible

    @classmethod
    def validate_artifacts(
        cls,
        root: str | Path,
        *,
        task: str,
        instance_id: int,
        num_rollouts: int | None = None,
        require_videos: bool | None = None,
        rollout_specs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[tuple[Path, Path]]:
        specs = cls._normalize_rollout_specs(
            num_rollouts=num_rollouts,
            require_videos=require_videos,
            rollout_specs=rollout_specs,
        )
        root = Path(root)
        json_dir = root / "json"
        video_dir = root / "videos"
        expected_json = {
            json_dir / f"{task}_{instance_id}_{spec['rollout_id']}.json"
            for spec in specs
        }
        actual_json = (
            set(json_dir.glob(f"{task}_{instance_id}_*.json"))
            if json_dir.is_dir()
            else set()
        )
        if actual_json != expected_json:
            missing = sorted(str(path) for path in expected_json - actual_json)
            unexpected = sorted(str(path) for path in actual_json - expected_json)
            raise ArtifactValidationError(
                f"rollout JSON set mismatch: missing={missing}, unexpected={unexpected}"
            )
        artifacts = []
        results = {}
        for spec in specs:
            rollout_id = spec["rollout_id"]
            path = json_dir / f"{task}_{instance_id}_{rollout_id}.json"
            try:
                value = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactValidationError(
                    f"invalid rollout JSON {path}: {exc}"
                ) from exc
            cls._validate_result(value, task, instance_id, rollout_id)
            if "sample_id" in spec and value.get("sample_id") != spec["sample_id"]:
                raise ArtifactValidationError(
                    "rollout result sample_id does not match "
                    f"expected value {spec['sample_id']!r}"
                )
            results[rollout_id] = value
            artifacts.append((path, Path("json") / path.name))

        expected_video = {
            video_dir / f"{task}_{instance_id}_{spec['rollout_id']}.mp4"
            for spec in specs
        }
        required_video = set()
        for spec in specs:
            rollout_id = spec["rollout_id"]
            if spec["video"] == VIDEO_REQUIRED or (
                spec["video"] == VIDEO_REQUIRED_IF_ELIGIBLE
                and cls._result_is_metric_eligible(results[rollout_id])
            ):
                required_video.add(
                    video_dir / f"{task}_{instance_id}_{rollout_id}.mp4"
                )
        actual_video = (
            set(video_dir.glob(f"{task}_{instance_id}_*.mp4"))
            if video_dir.is_dir()
            else set()
        )
        missing = sorted(str(path) for path in required_video - actual_video)
        unexpected = sorted(str(path) for path in actual_video - expected_video)
        if missing or unexpected:
            raise ArtifactValidationError(
                f"rollout video set mismatch: missing={missing}, unexpected={unexpected}"
            )
        for path in sorted(actual_video):
            if path.stat().st_size <= 0:
                raise ArtifactValidationError(f"rollout video is empty: {path}")
            artifacts.append((path, Path("videos") / path.name))
        # A rollout JSON is the completion signal consumed by progress tooling.
        # Publish videos first so a partial publication cannot expose a completed JSON early.
        artifacts.sort(key=lambda item: item[1].parts[0] != "videos")
        return artifacts

    @staticmethod
    def _assert_same_artifact(source: Path, destination: Path) -> None:
        if source.samefile(destination):
            return
        if destination.stat().st_size != source.stat().st_size or _sha256(
            destination
        ) != _sha256(source):
            raise ArtifactValidationError(
                f"refusing to overwrite different artifact: {destination}"
            )

    @classmethod
    def _publish_one(cls, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            cls._assert_same_artifact(source, destination)
            return
        try:
            os.link(source, destination)
            return
        except FileExistsError:
            cls._assert_same_artifact(source, destination)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                cls._assert_same_artifact(source, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _mark_succeeded(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        lease: JobLease,
        provenance: dict | None,
    ) -> None:
        timestamp = _now()
        provenance_value = {
            "attempt": lease.attempt,
            "worker": lease.worker,
            "lease_token": lease.lease_token,
            **(provenance or {}),
        }
        connection.execute(
            """
            UPDATE jobs
            SET state = 'succeeded', succeeded_at = ?, updated_at = ?, provenance = ?
            WHERE id = ?
            """,
            (timestamp, timestamp, _json_text(provenance_value), row["id"]),
        )
        connection.execute(
            """
            UPDATE attempts SET state = 'succeeded', ended_at = ?
            WHERE job_id = ? AND attempt = ? AND lease_token = ? AND state = 'running'
            """,
            (timestamp, row["id"], lease.attempt, lease.lease_token),
        )

    def publish(
        self,
        lease: JobLease,
        canonical_task_dir: str | Path,
        *,
        num_rollouts: int | None = None,
        require_videos: bool | None = None,
        rollout_specs: Sequence[Mapping[str, object]] | None = None,
        provenance: dict | None = None,
    ) -> None:
        canonical_task_dir = Path(canonical_task_dir).expanduser().resolve()
        validation = {
            "num_rollouts": num_rollouts,
            "require_videos": require_videos,
            "rollout_specs": rollout_specs,
        }
        staged = self.validate_artifacts(
            lease.attempt_dir,
            task=lease.task,
            instance_id=lease.instance_id,
            **validation,
        )
        with self._immediate() as connection:
            row = self._assert_lease(connection, lease)
            for source, relative in staged:
                self._publish_one(source, canonical_task_dir / relative)
            self.validate_artifacts(
                canonical_task_dir,
                task=lease.task,
                instance_id=lease.instance_id,
                **validation,
            )
            self._mark_succeeded(connection, row, lease, provenance)

    def reconcile(
        self,
        lease: JobLease,
        canonical_task_dir: str | Path,
        *,
        num_rollouts: int | None = None,
        require_videos: bool | None = None,
        rollout_specs: Sequence[Mapping[str, object]] | None = None,
        provenance: dict | None = None,
    ) -> bool:
        canonical_task_dir = Path(canonical_task_dir).expanduser().resolve()
        validation = {
            "num_rollouts": num_rollouts,
            "require_videos": require_videos,
            "rollout_specs": rollout_specs,
        }
        try:
            self.validate_artifacts(
                canonical_task_dir,
                task=lease.task,
                instance_id=lease.instance_id,
                **validation,
            )
        except ArtifactValidationError:
            try:
                self.validate_artifacts(
                    lease.attempt_dir,
                    task=lease.task,
                    instance_id=lease.instance_id,
                    **validation,
                )
            except ArtifactValidationError:
                return False
            self.publish(
                lease,
                canonical_task_dir,
                **validation,
                provenance={"reconciled": True, **(provenance or {})},
            )
            return True
        with self._immediate() as connection:
            row = self._assert_lease(connection, lease)
            self._mark_succeeded(
                connection,
                row,
                lease,
                {"reconciled": True, **(provenance or {})},
            )
        return True

    def counts(self, *, readonly: bool = False) -> dict:
        with self._connect(readonly=readonly) as connection:
            counts = {state: 0 for state in JOB_STATES}
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            ):
                counts[row["state"]] = row["count"]
        return {"counts": counts, "total": sum(counts.values())}

    def snapshot(self, *, readonly: bool = False) -> dict:
        with self._connect(readonly=readonly) as connection:
            counts = {state: 0 for state in JOB_STATES}
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            ):
                counts[row["state"]] = row["count"]
            rows = connection.execute(
                """
                SELECT task, instance_id, state, attempts, max_attempts, worker, lease_token,
                    attempt_dir, last_error, succeeded_at, provenance
                FROM jobs ORDER BY task, instance_id
                """
            ).fetchall()
            attempt_rows = connection.execute(
                """
                SELECT jobs.task, jobs.instance_id, attempts.attempt, attempts.worker,
                    attempts.lease_token, attempts.attempt_dir, attempts.state,
                    attempts.started_at, attempts.ended_at, attempts.error
                FROM attempts JOIN jobs ON jobs.id = attempts.job_id
                ORDER BY jobs.task, jobs.instance_id, attempts.attempt
                """
            ).fetchall()
        jobs = []
        for row in rows:
            value = dict(row)
            value["provenance"] = (
                json.loads(value["provenance"]) if value["provenance"] else None
            )
            jobs.append(value)
        return {
            "counts": counts,
            "total": len(jobs),
            "jobs": jobs,
            "attempts": [dict(row) for row in attempt_rows],
        }

    def create_result_manifest(
        self,
        results_root: str | Path,
        output_path: str | Path,
        *,
        model: str,
        num_rollouts: int | None = None,
        require_videos: bool | None = None,
        rollout_specs_by_job: Mapping[
            tuple[str, int], int | Sequence[Mapping[str, object]]
        ]
        | None = None,
    ) -> dict:
        results_root = Path(results_root).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        snapshot = self.snapshot(readonly=True)
        incomplete = [job for job in snapshot["jobs"] if job["state"] != "succeeded"]
        if incomplete:
            raise RuntimeError(
                f"cannot create result manifest with {len(incomplete)} incomplete jobs"
            )

        job_keys = {(job["task"], job["instance_id"]) for job in snapshot["jobs"]}
        if rollout_specs_by_job is not None:
            provided_keys = set(rollout_specs_by_job)
            if provided_keys != job_keys:
                missing = sorted(job_keys - provided_keys)
                unexpected = sorted(provided_keys - job_keys, key=repr)
                raise ValueError(
                    "rollout_specs_by_job keys do not match queue jobs: "
                    f"missing={missing}, unexpected={unexpected}"
                )

        rollouts = []
        for job in snapshot["jobs"]:
            job_key = (job["task"], job["instance_id"])
            if rollout_specs_by_job is None:
                specs = self._normalize_rollout_specs(
                    num_rollouts=num_rollouts,
                    require_videos=require_videos,
                    rollout_specs=None,
                )
                validation = {
                    "num_rollouts": num_rollouts,
                    "require_videos": require_videos,
                }
            else:
                job_rollouts = rollout_specs_by_job[job_key]
                if isinstance(job_rollouts, int) and not isinstance(job_rollouts, bool):
                    specs = self._normalize_rollout_specs(
                        num_rollouts=job_rollouts,
                        require_videos=require_videos,
                        rollout_specs=None,
                    )
                    validation = {
                        "num_rollouts": job_rollouts,
                        "require_videos": require_videos,
                    }
                else:
                    specs = self._normalize_rollout_specs(
                        num_rollouts=None,
                        require_videos=None,
                        rollout_specs=job_rollouts,
                    )
                    validation = {"rollout_specs": job_rollouts}

            task_dir = results_root / model / job["task"]
            self.validate_artifacts(
                task_dir,
                task=job["task"],
                instance_id=job["instance_id"],
                **validation,
            )
            for spec in specs:
                rollout_id = spec["rollout_id"]
                stem = f"{job['task']}_{job['instance_id']}_{rollout_id}"
                json_path = task_dir / "json" / f"{stem}.json"
                result = json.loads(json_path.read_text())
                record = {
                    "task": job["task"],
                    "instance_id": job["instance_id"],
                    "rollout_id": rollout_id,
                    "json": _artifact(json_path, results_root),
                    "provenance": job["provenance"],
                }
                if "sample_id" in result:
                    record["sample_id"] = result["sample_id"]
                video_path = task_dir / "videos" / f"{stem}.mp4"
                record["video"] = (
                    _artifact(video_path, results_root) if video_path.exists() else None
                )
                rollouts.append(record)
        rollouts.sort(
            key=lambda item: (item["task"], item["instance_id"], item["rollout_id"])
        )
        manifest = {
            "schema_version": 1,
            "metadata": self.metadata(readonly=True),
            "rollout_count": len(rollouts),
            "rollouts": rollouts,
        }
        atomic_write_json(output_path, manifest)
        return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a pi05 SQLite evaluation queue."
    )
    parser.add_argument("db", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            EvaluationQueue(args.db).snapshot(readonly=True), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    _main()
