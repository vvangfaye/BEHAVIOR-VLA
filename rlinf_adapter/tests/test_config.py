from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_composed_config_preserves_policy_contract(monkeypatch):
    from conftest import ADAPTER, UPSTREAM

    for key, value in {
        "RLINF_ROOT": str(UPSTREAM),
        "EMBODIED_PATH": str(UPSTREAM / "examples/embodiment"),
        "B1K_RL_CHECKPOINT": "/weights",
        "B1K_TOKENIZER": "/tokenizer.model",
        "B1K_RL_OUTPUT": "/results",
        "B1K_TRAIN_INSTANCES": "/train",
        "B1K_EVAL_INSTANCES": "/eval",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(ADAPTER / "config"), version_base="1.1"):
        cfg = compose(config_name="behavior_joint_pi05")
    OmegaConf.resolve(cfg)
    assert cfg.actor.model.openpi.action_horizon == 32
    assert cfg.actor.model.num_action_chunks == 16
    assert cfg.actor.model.openpi.max_token_len == 512
    assert cfg.actor.model.action_dim == 23
    assert cfg.actor.model.b1k_joint.state_layout == "canonical25"
    assert cfg.actor.model.b1k_joint.execution_gripper_open_tolerance == 0.0
    assert OmegaConf.to_container(cfg.rollout.model, resolve=True) == OmegaConf.to_container(
        cfg.actor.model, resolve=True
    )
    assert cfg.actor.model.openpi.train_expert_only
    assert (
        cfg.env.train.omni_config.task.activity_instance_id
        != cfg.env.eval.omni_config.task.activity_instance_id
    )
