# ruff: noqa: SLF001 -- regression-test the legacy private renderer directly.
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_SPEC = importlib.util.spec_from_file_location(
    "gen_subtask_prompts", Path(__file__).with_name("gen_subtask_prompts.py")
)
assert _SPEC is not None
assert _SPEC.loader is not None
prompts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prompts)


def _segment(description, manipulating, objects, start=0, end=1, skill_idx=0, **extra):
    return {
        "skill_idx": skill_idx,
        "skill_description": [description],
        "manipulating_object_id": manipulating,
        "object_id": [objects],
        "memory_prefix": [],
        "spatial_prefix": [],
        "frame_duration": [start, end],
        **extra,
    }


def test_role_aware_grocery_labels_follow_annotation_roles():
    segments = [
        _segment("move to", [], ["grocery_sack_7"]),
        _segment("pick up from", ["tomato_2"], ["tomato_2", "grocery_sack_7"]),
        _segment("place in", ["tomato_2"], ["tomato_2", "refrigerator_xylabc_0"]),
        _segment("close lid", ["car_4"], ["car_4"]),
    ]
    aliases = prompts.stable_object_aliases(segments)

    assert prompts.role_aware_subtask_text(segments[0], aliases) == "move to grocery sack"
    assert prompts.role_aware_subtask_text(segments[1], aliases) == "pick up tomato from grocery sack"
    assert prompts.role_aware_subtask_text(segments[2], aliases) == "place tomato in refrigerator"
    assert prompts.role_aware_subtask_text(segments[3], aliases) == "close lid of car"
    assert "target to reference" not in " ".join(prompts.role_aware_subtask_text(seg, aliases) for seg in segments)

    semantic_suffix = _segment(
        "place in",
        ["beefsteak_tomato_80"],
        ["beefsteak_tomato_80", "fridge_petcxr_0"],
    )
    aliases = prompts.stable_object_aliases([semantic_suffix])
    assert prompts.role_aware_subtask_text(semantic_suffix, aliases) == "place beefsteak tomato in fridge"
    close_car = _segment("close lid", ["car_ssxsje_0"], ["car_ssxsje_0"])
    aliases = prompts.stable_object_aliases([close_car])
    assert prompts.role_aware_subtask_text(close_car, aliases) == "close lid of car"


def test_role_aware_rose_aliases_require_distinct_raw_ids_and_normalize_prefixes():
    one_rose = [
        _segment(
            "move to",
            [],
            ["rose"],
            memory_prefix=["remember_exactly"],
            spatial_prefix=["left_of_vase"],
        ),
        _segment("place in", ["rose"], ["rose", "vase_1"]),
    ]
    one_alias = prompts.stable_object_aliases(one_rose)
    assert one_alias["rose"] == "rose"
    assert prompts.role_aware_subtask_text(one_rose[0], one_alias) == ("remember exactly left of vase move to rose")
    assert prompts.role_aware_subtask_text(one_rose[1], one_alias) == "place rose in vase"

    distinct_roses = [
        _segment("pick up from", ["rose_1"], ["rose_1", "table_1"]),
        _segment("pick up from", ["rose_2"], ["rose_2", "table_1"]),
    ]
    aliases = prompts.stable_object_aliases(distinct_roses)
    assert aliases["rose_1"] == "rose 1"
    assert aliases["rose_2"] == "rose 2"
    assert prompts.role_aware_subtask_text(distinct_roses[1], aliases) == "pick up rose 2 from table"


def test_role_aware_segments_preserve_boundaries_and_provenance():
    annotation = {
        "skill_annotation": [
            _segment("move to", [], ["rose"], start=1, end=3, skill_idx=10),
            _segment("move to", [], ["rose"], start=3, end=5, skill_idx=11),
            _segment("place in", ["rose"], ["rose", "vase_1"], start=7, end=8, skill_idx=12),
        ]
    }

    rows, uncovered, vocabulary = prompts.build_role_aware_segments(
        annotation, episode_index=42, length=10, annotation_path="annotations/task-0056/episode.json"
    )

    assert [(row["start"], row["end"], row["provenance"]) for row in rows] == [
        (0, 1, "imputed_leading"),
        (1, 3, "annotated"),
        (3, 5, "annotated"),
        (5, 7, "imputed_gap"),
        (7, 8, "annotated"),
        (8, 10, "imputed_tail"),
    ]
    assert [row["source_segment_index"] for row in rows] == [0, 0, 1, 1, 2, 2]
    assert [row["source_skill_index"] for row in rows] == [10, 10, 11, 11, 12, 12]
    assert rows[0]["subtask"] == "move to rose"
    assert rows[4]["manipulating_object_ids"] == ["rose"]
    assert rows[4]["reference_object_ids"] == ["vase_1"]
    assert rows[4]["object_ids"] == ["rose", "vase_1"]
    assert rows[4]["annotation_path"] == "annotations/task-0056/episode.json"
    assert uncovered == 5
    assert vocabulary == {"move to rose", "place rose in vase"}


def test_legacy_text_and_rle_regression():
    segment = _segment(
        "pick up from",
        ["radio_89"],
        ["radio_89", "coffee_table_koagbh_0"],
        start=1,
        end=3,
    )
    annotation = {"skill_annotation": [segment, {**segment, "skill_idx": 1, "frame_duration": [3, 4]}]}

    assert prompts._subtask_text(segment) == "pick up from radio to coffee table koagbh"
    rows, uncovered, vocabulary = prompts.build_legacy_segments(annotation, 7, 6, "skill_annotation")
    assert rows == [
        {
            "episode_index": 7,
            "start": 0,
            "end": 6,
            "subtask": "pick up from radio to coffee table koagbh",
        },
    ]
    assert uncovered == 3
    assert vocabulary == {"pick up from radio to coffee table koagbh"}


def test_cli_defaults_to_legacy_four_column_output(tmp_path, monkeypatch):
    annotation_path = "annotations/task-0000/episode.json"
    metadata_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    metadata_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [7], "length": [3], "annotation_path": [annotation_path]}),
        metadata_dir / "file-000.parquet",
    )
    annotation_file = tmp_path / annotation_path
    annotation_file.parent.mkdir(parents=True)
    annotation_file.write_text(
        json.dumps(
            {"skill_annotation": [_segment("pick up from", ["radio_89"], ["radio_89", "coffee_table_koagbh_0"], end=2)]}
        )
    )
    output = tmp_path / "segments.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_subtask_prompts.py", "--dataset", str(tmp_path), "--out", str(output)],
    )

    prompts.main()

    table = pq.read_table(output)
    assert table.column_names == ["episode_index", "start", "end", "subtask"]
    assert table.to_pylist() == [
        {
            "episode_index": 7,
            "start": 0,
            "end": 3,
            "subtask": "pick up from radio to coffee table koagbh",
        }
    ]


def test_episode_metadata_can_filter_task_indices(tmp_path):
    metadata_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    metadata_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [7, 8],
                "length": [3, 4],
                "annotation_path": ["annotations/a.json", "annotations/b.json"],
                "task_index": [14, 56],
            }
        ),
        metadata_dir / "file-000.parquet",
    )

    assert prompts.load_episode_meta(str(tmp_path), {56}) == [(8, 4, "annotations/b.json")]


def test_role_aware_cli_marks_imputed_rows_unannotated(tmp_path, monkeypatch):
    annotation_path = "annotations/task-0000/episode.json"
    metadata_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    metadata_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [7], "length": [4], "annotation_path": [annotation_path]}),
        metadata_dir / "file-000.parquet",
    )
    annotation_file = tmp_path / annotation_path
    annotation_file.parent.mkdir(parents=True)
    annotation_file.write_text(json.dumps({"skill_annotation": [_segment("move to", [], ["rose"], start=1, end=3)]}))
    output = tmp_path / "segments-role-aware.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gen_subtask_prompts.py",
            "--dataset",
            str(tmp_path),
            "--out",
            str(output),
            "--label-format",
            "role_aware_v1",
        ],
    )

    prompts.main()

    rows = pq.read_table(output).to_pylist()
    assert [(row["start"], row["end"], row["is_annotated"], row["provenance"]) for row in rows] == [
        (0, 1, False, "imputed_leading"),
        (1, 3, True, "annotated"),
        (3, 4, False, "imputed_tail"),
    ]
