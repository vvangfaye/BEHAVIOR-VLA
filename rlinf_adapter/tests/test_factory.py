from omegaconf import OmegaConf

from b1k_rlinf.factory import build_action_interface, build_rl_config


def test_top_level_value_head_switch_reaches_rl_model():
    cfg = OmegaConf.create({"add_value_head": True, "openpi": {"value_after_vlm": True}})
    assert build_rl_config(cfg).add_value_head
    cfg.add_value_head = False
    assert not build_rl_config(cfg).add_value_head


def test_action_interface_is_opt_in_and_explicit_in_configuration():
    assert build_action_interface(OmegaConf.create({})).gripper_open_tolerance == 0
    cfg = OmegaConf.create({"b1k_joint": {"execution_gripper_open_tolerance": 0.001}})
    assert build_action_interface(cfg).gripper_open_tolerance == 0.001
