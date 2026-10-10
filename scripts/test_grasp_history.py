"""Check preparation changes only native grasp fields and remains non-scoring."""
import ast
import copy
from pathlib import Path
import pytest


def carry():
    path=Path(__file__).with_name('diagnose_b1k_grasp_history.py')
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='carry_native_grasp')
    scope={'copy':copy};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['carry_native_grasp']


def fixture():
    return {0:{'registry':{'object_registry':{'robot':{'controller_groups':{'arm':{'goal':[1,2]}},'joint_pos':[3,4],'ag_obj_constraint_params':{}},'basket':{'root_link':{'pos':[5,6,7]}}},'system_registry':{'water':{'particles':[8,9]}}}}}


def test_history_preserves_all_recorded_fields_and_does_not_mutate_inputs():
    source=fixture();original=copy.deepcopy(source)
    native={'ag_obj_constraint_params':{'left':{'target_obj':'basket','joint_type':'FixedJoint','parent_frame_pos':[.1,.2,.3]}}}
    result=carry()(source,'robot',native)
    expected=copy.deepcopy(source);expected[0]['registry']['object_registry']['robot']['ag_obj_constraint_params']=copy.deepcopy(native['ag_obj_constraint_params'])
    assert result==expected and source==original
    result[0]['registry']['object_registry']['robot']['ag_obj_constraint_params']['left']['parent_frame_pos'][0]=10
    assert native['ag_obj_constraint_params']['left']['parent_frame_pos'][0]==.1


def test_history_does_not_keep_a_grasp_after_the_physical_prefix_releases_it():
    source=fixture();source[0]['registry']['object_registry']['robot']['ag_obj_constraint_params']={'left':{'target_obj':'basket'}}
    result=carry()(source,'robot',{'ag_obj_constraint_params':{}})
    assert result[0]['registry']['object_registry']['robot']['ag_obj_constraint_params']=={}


@pytest.mark.parametrize('damage',['extra_scene','missing_robot','undecoded_robot','missing_native','unsupported_arm'])
def test_history_refuses_unknown_scene_or_robot_grasp_state(damage):
    source=fixture();native={'ag_obj_constraint_params':{}}
    if damage=='extra_scene':source[1]={}
    elif damage=='missing_robot':source[0]['registry']['object_registry'].pop('robot')
    elif damage=='undecoded_robot':source[0]['registry']['object_registry']['robot'].pop('controller_groups')
    elif damage=='missing_native':native={}
    else:native['ag_obj_constraint_params']={'third':{}}
    with pytest.raises(ValueError):carry()(source,'robot',native)
