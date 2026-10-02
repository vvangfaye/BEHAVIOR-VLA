"""Command entry point. Heavy libraries load only for the selected command."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="GoalKeeper research prototype for BEHAVIOR 2026")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_cmd = commands.add_parser("compile", help="Compile a PUBLIC task definition")
    compile_cmd.add_argument("bddl")
    audit = commands.add_parser("audit", help="Report supported BDDL coverage; fail if any unsupported")
    audit.add_argument("directory")
    synthetic = commands.add_parser("synthetic", help="Generate software-only toy data")
    synthetic.add_argument("output"); synthetic.add_argument("--groups", type=int, default=120)
    synthetic.add_argument("--windows", type=int, default=3)
    train_cmd = commands.add_parser("train", help="Train the small goal head, not Comet")
    train_cmd.add_argument("manifest"); train_cmd.add_argument("output")
    train_cmd.add_argument("--epochs", type=int, default=20)
    train_cmd.add_argument("--batch-size", "--global-batch-size", dest="batch_size", type=int, default=16,
                           help="Global batch size; must divide WORLD_SIZE")
    train_cmd.add_argument("--learning-rate", type=float, default=3e-4)
    train_cmd.add_argument("--hidden-dim", type=int, default=128)
    train_cmd.add_argument("--device", default="auto")
    train_cmd.add_argument("--workers", type=int, default=0)
    train_cmd.add_argument("--dataset-contract", help="Explicit dataset_contract.json (required for real data unless adjacent)")
    train_cmd.add_argument("--seed", type=int, default=7)
    cal = commands.add_parser("calibrate", help="Calibrate on independent initial-state groups")
    cal.add_argument("manifest"); cal.add_argument("checkpoint"); cal.add_argument("output")
    cal.add_argument("--device", default="cpu"); cal.add_argument("--min-count", type=int, default=30)
    cal.add_argument("--dataset-contract", help="Explicit dataset_contract.json (required for real data unless adjacent)")
    evaluate = commands.add_parser("report", help="Aggregate original official metrics JSON")
    evaluate.add_argument("results"); evaluate.add_argument("expected", help="JSON array of [task,instance_id,rollout_id]")
    evaluate.add_argument("--baseline"); evaluate.add_argument("--output")
    export = commands.add_parser("export", help="Export OFFLINE replayed training episode features")
    export.add_argument("episode"); export.add_argument("bddl"); export.add_argument("output")
    export.add_argument("--group-id", required=True); export.add_argument("--action-spec", required=True)
    export.add_argument("--encoder", required=True); export.add_argument("--revision", required=True)
    export.add_argument("--device", default="cuda:0"); export.add_argument("--horizon", type=int, default=16)
    export.add_argument("--history", type=int, default=4)
    export_batch = commands.add_parser("export-batch", help="Export and seal an explicit batch of training episodes")
    export_batch.add_argument("spec"); export_batch.add_argument("output")
    export_batch.add_argument("--encoder", required=True); export_batch.add_argument("--revision", required=True)
    export_batch.add_argument("--device", default="cuda:0"); export_batch.add_argument("--horizon", type=int, default=16)
    export_batch.add_argument("--history", type=int, default=4)
    merge = commands.add_parser("merge", help="Merge feature manifests into a sealed formal dataset")
    merge.add_argument("output"); merge.add_argument("manifests", nargs="+")
    check = commands.add_parser("check", help="Verify a sealed dataset and all content hashes")
    check.add_argument("dataset")
    serve = commands.add_parser("serve", help="Serve raw OpenPI + optional trained goal head")
    serve.add_argument("config", help="Explicit public task + model configuration JSON")
    args = parser.parse_args(argv)
    options = vars(args).copy(); command = options.pop("command")
    if command in ("compile", "audit"):
        from .goals import compile_bddl
        if command == "compile": result = compile_bddl(Path(args.bddl).read_text()).as_dict()
        else:
            files = sorted(Path(args.directory).rglob("*.bddl"))
            if not files: raise ValueError("No .bddl files found")
            result = {"total": len(files), "supported": [], "unsupported": {}}
            for path in files:
                try:
                    goal = compile_bddl(path.read_text())
                    result["supported"].append({"path": str(path), "atoms": len(goal.atoms), "clauses": len(goal.clauses)})
                except ValueError as error: result["unsupported"][str(path)] = str(error)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["unsupported"] else 0
    elif command == "synthetic":
        from .synthetic import make_dataset
        result = {"manifest": make_dataset(**options), "warning": "SYNTHETIC DATA; no BEHAVIOR result"}
    elif command == "train":
        from .training import train
        train(**options); return 0
    elif command == "calibrate":
        from .training import calibrate
        result = calibrate(**options)
    elif command == "report":
        from .evaluation import summarize
        result = summarize(args.results, json.loads(Path(args.expected).read_text()), args.baseline)
        if args.output: Path(args.output).write_text(json.dumps(result, indent=2))
    elif command == "export":
        from .prepare import export_episode
        from .features import SiglipFeatures
        options.pop("encoder"); options.pop("revision"); options.pop("device")
        result = {"manifest": export_episode(**options, features=SiglipFeatures(args.encoder, args.revision, args.device))}
    elif command == "export-batch":
        from .prepare import export_batch
        from .features import SiglipFeatures
        options.pop("encoder"); options.pop("revision"); options.pop("device")
        result = {"manifest": export_batch(**options, features=SiglipFeatures(args.encoder, args.revision, args.device))}
    elif command == "merge":
        from .data import merge_manifests, check_dataset
        manifest = merge_manifests(args.manifests, args.output)
        result = {"manifest": manifest, "contract": check_dataset(Path(manifest).parent)}
    elif command == "check":
        from .data import check_dataset
        result = check_dataset(args.dataset)
    else:
        serve_config(args.config); return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def serve_config(path):
    import copy
    import dataclasses
    from .goals import compile_bddl, GoalProgram, UnsupportedGoal, parse_sexpr
    from .policy import GoalkeeperPolicy, JointSubtaskChunkSampler, LearnedPredictor, OpenPIChunkSampler
    from .observations import SensorSchema
    from .features import SiglipFeatures
    from .belief import PreservationSelector
    from .server import serve_forever
    from openpi.configs.robots import ROBOT_REGISTRY
    from openpi.configs.tasks import TASK_REGISTRY
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    settings = json.loads(Path(path).read_text())
    for key in ("task_id", "proprio_dim", "action_dim", "horizon"):
        if type(settings.get(key)) is not int: raise ValueError(f"Fill the actual integer {key} in your configuration")
    expected_task = os.environ.get("GOALKEEPER_EXPECTED_TASK")
    if expected_task and settings.get("task") != expected_task:
        raise ValueError("GoalKeeper config task disagrees with the evaluator task")
    task_bucket, task_name = settings["task"].split("/", 1)
    instruction = settings.get("instruction") or TASK_REGISTRY[task_bucket][task_name]
    robot = ROBOT_REGISTRY[settings["robot"]]
    if settings["action_dim"] != robot.action_dim:
        raise ValueError("Configured action_dim disagrees with the OpenPI robot configuration")
    camera_keys = sorted(robot.observations)
    if len(camera_keys) != 3 or any(not robot.observations[key].obs_key.endswith("::rgb") for key in camera_keys):
        raise ValueError("GoalKeeper currently requires exactly three RGB cameras")
    camera_names = [robot.observations[key].obs_key.removesuffix("::rgb") for key in camera_keys]
    schema = SensorSchema(robot=robot.name, head=camera_names[0], left=camera_names[1], right=camera_names[2])
    mode = settings.get("mode", "baseline")
    public_text = Path(settings["bddl"]).read_text()
    try:
        program = compile_bddl(public_text)
    except UnsupportedGoal as error:
        if mode != "baseline": raise
        # An explicit baseline run needs no goal interpretation. Keep a hash for
        # provenance; do not silently weaken unsupported goals in rerank mode.
        import hashlib, logging
        parse_sexpr(public_text)
        logging.warning("Explicit baseline mode: goal predictor disabled: %s", error)
        program = GoalProgram("uncompiled_public_task", {}, (), (), hashlib.sha256(public_text.encode()).hexdigest())
    config = openpi_config.get_config(settings["openpi_config"])
    joint = settings.get("subtask_prediction", getattr(config.model, "train_subtask_prediction", False))
    if type(joint) is not bool:
        raise ValueError("subtask_prediction must be a boolean")
    if not joint and getattr(config.model, "train_subtask_prediction", False):
        raise ValueError("subtask_prediction=false disagrees with the selected joint OpenPI config")
    if joint:
        max_token_len = settings.get("max_token_len", 512)
        max_subtask_len = settings.get("max_subtask_len", 64)
        if any(type(value) is not int or value < 1 for value in (max_token_len, max_subtask_len)):
            raise ValueError("max_token_len and max_subtask_len must be positive integers")
        config = dataclasses.replace(config, model=dataclasses.replace(
            config.model, train_subtask_prediction=True,
            max_token_len=max_token_len, max_subtask_len=max_subtask_len,
            subtask_loss_chunk_size=min(config.model.subtask_loss_chunk_size, max_subtask_len)))
    if settings["horizon"] > config.model.action_horizon:
        raise ValueError("GoalKeeper execution horizon exceeds the OpenPI model action horizon")
    config = dataclasses.replace(config, data=dataclasses.replace(
        config.data, repo_id=settings["repo_id"], robot_config_name=settings["robot"]))
    base = policy_config.create_trained_policy(
        config, settings["openpi_checkpoint"], default_prompt=instruction,
        subtask_interval=settings.get("subtask_interval"),
        subtask_temperature=settings.get("subtask_temperature", 0.0),
        subtask_action_conditioning=settings.get("subtask_action_conditioning"))
    if joint and not hasattr(base, "infer_candidates"):
        raise TypeError("Joint GoalKeeper requires a SubtaskPolicy with infer_candidates")
    if base.metadata.get("framesamp_modul", False):
        raise ValueError("GoalKeeper does not yet forward every control frame to FrameSamp memory")
    features = None
    if mode != "baseline":
        features = SiglipFeatures(settings["encoder"], settings["encoder_revision"], settings.get("device", "cuda:0"))
    def factory():
        predictor = None
        if features:
            predictor = LearnedPredictor(program, settings["head_checkpoint"], settings["calibration"],
                                         features, settings["action_spec"], settings.get("device", "cuda:0"))
        sampler_type = JointSubtaskChunkSampler if joint else OpenPIChunkSampler
        # Share read-only model weights, but keep RNG and generated-subtask
        # cache in a distinct Policy instance for each websocket connection.
        session_base = copy.copy(base)
        return GoalkeeperPolicy(program, sampler_type(session_base, instruction),
                                task_id=settings["task_id"], proprio_dim=settings["proprio_dim"],
                                action_dim=settings["action_dim"], horizon=settings["horizon"],
                                candidates=settings.get("candidates", 4), mode=mode, predictor=predictor,
                                schema=schema,
                                selector=PreservationSelector(**settings.get("selector", {})))
    # Each process serves one public task. Run separate processes for different
    # tasks/checkpoints. The websocket loop serializes inference in this prototype.
    serve_forever(factory, settings.get("host", "127.0.0.1"), settings.get("port", 8000),
                  server_token=os.environ.get("GOALKEEPER_SERVER_TOKEN"))


if __name__ == "__main__":
    raise SystemExit(main())
