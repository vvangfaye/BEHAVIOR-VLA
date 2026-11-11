"""Audit the frozen input projection and limits of feedback identification.

No models are run. NumPy is used only for local cached observations.
"""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def indices(node):
    if isinstance(node, ast.List):
        return [ast.literal_eval(value) for value in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'list' and len(node.args) == 1):
        call = node.args[0]
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'range':
            return list(range(*(ast.literal_eval(value) for value in call.args)))
    raise ValueError('Unexpected frozen config expression')


def main():
    bundle = ROOT/'pilot_cloud_bundle.tar.gz'
    wanted = {
        'config': 'project/BEHAVIOR-VLAs/openpi/src/openpi/configs/robots/b1k.py',
        'projection': 'project/BEHAVIOR-VLAs/openpi/src/openpi/policies/b1k_policy.py',
    }
    with tarfile.open(bundle, 'r:gz') as archive:
        sources = {name: archive.extractfile(member).read() for name, member in wanted.items()}
    config_ast = ast.parse(sources['config'])
    call = next(node.value for node in config_ast.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'R1Pro' for t in node.targets))
    components = []
    for item in next(kw.value for kw in call.keywords if kw.arg == 'proprio').elts:
        fields = {kw.arg: kw.value for kw in item.keywords}
        components.append(SimpleNamespace(name=ast.literal_eval(fields['name']), indices=indices(fields['indices']),
                                          is_eef=ast.literal_eval(fields.get('is_eef', ast.Constant(False)))))
    function = next(node for node in ast.parse(sources['projection']).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'extract_state_from_proprio')
    namespace = {'np': np, 'RobotConfig': object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen_extract_state>', 'exec'), namespace)
    extract = namespace['extract_state_from_proprio']
    config = SimpleNamespace(proprio=components)
    selected = {index for component in components for index in component.indices}
    discarded = sorted(set(range(61)) - selected)
    velocity_groups = {'left_arm_velocity': list(range(10, 17)),
                       'left_finger_velocity': list(range(26, 28)),
                       'right_arm_velocity': list(range(35, 42)),
                       'right_finger_velocity': list(range(51, 53)),
                       'trunk_velocity': list(range(57, 61))}
    assert all(set(value).isdisjoint(selected) for value in velocity_groups.values())
    cache = ROOT/'pilot_three_tasks_cache_v1'
    manifest = json.loads((cache/'cache_manifest.json').read_text())
    observations = []
    projection_checks = []
    for item in manifest['records']:
        path = cache/item['npz_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['npz_sha256']
        with np.load(path, allow_pickle=False) as data:
            state = data['observation.state'].copy()
        assert state.shape == (61,) and np.isfinite(state).all()
        perturbed = state.copy()
        perturbed[discarded] += np.float32(.001)
        current, changed = extract(state, config), extract(perturbed, config)
        assert current.shape == (23,) and np.array_equal(current, changed)
        projection_checks.append({'sample_id': item['sample_id'], 'identical_projected_state': True})
        observations.append(state)
    observations = np.stack(observations)
    velocity_summary = {
        name: {'indices': group,
               'coordinates': [{'raw_index': idx,
                                'nonzero_observations_above_1e_6': int(np.count_nonzero(np.abs(observations[:,idx])>1e-6)),
                                'rms_in_coordinate_units': float(np.sqrt(np.mean(observations[:,idx].astype(float)**2)))}
                               for idx in group]}
        for name, group in velocity_groups.items()
    }
    # Unknown intended reference makes observed feedback gain non-identifiable.
    q, u = np.array([-.3, .2, .5]), np.array([.4, -.1, .2])
    examples = []
    for gain in [.5, 1., 3.]:
        reference = u + gain*q
        assert np.allclose(reference-gain*q, u)
        examples.append({'gain': gain, 'compatible_reference': reference.tolist()})
    report = {
        'status': 'complete', 'source_bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
        'frozen_sources': {name: {'member': wanted[name], 'sha256': hashlib.sha256(content).hexdigest()}
                           for name, content in sources.items()},
        'input_projection': {'raw_dimension': 61, 'projected_dimension': 23,
                             'components': [vars(c) for c in components],
                             'selected_raw_coordinates': sorted(selected), 'discarded_raw_coordinates': discarded,
                             'discarded_velocity_coordinates': sum(len(group) for group in velocity_groups.values()),
                             'base_velocity_is_retained': True,
                             'other_discarded_information': 'Two end-effector poses; individual finger positions are summed, additionally losing finger asymmetry.'},
        'executed_checks': {'samples': len(observations), 'discarded_field_changes_leave_projected_state_identical': True,
                            'per_sample': projection_checks},
        'velocity_summary': velocity_summary,
        'tracking_error_identification_limit': {
            'model': 'u_t = r_t - k*q_t', 'observed_q': q.tolist(), 'observed_u': u.tolist(),
            'compatible_gains_and_references': examples,
            'conclusion': 'Unknown intended reference permits multiple feedback gains consistent with exactly the same recorded position/action pairs. Tracking errors alone do not identify human corrective feedback.',
            'not_excluded': 'Identification under additional validated reference, disturbance, excitation or model assumptions.'},
        'research_interpretation': [
            'The raw data contain motion signals unused by this input projection; no new human data are required to test their usefulness.',
            'This is not evidence that omitted velocity is needed by the high-level policy; the position controller may suppress its effect.',
            'Raw field blindness is not proof of blindness to velocity inferred from history or visual cues.',
            'Adding velocity or history alone is not a novel training paradigm.',
            'A physical experiment must show task-relevant action or outcome differences at otherwise matched observations before claiming harmful state aliasing.'
        ],
        'new_model_calls': 0, 'new_simulation': False, 'goal_complete': False,
    }
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'raw_dimension':61, 'projected_dimension':23, 'samples_verified':len(observations),
                      'discarded_velocity_dimensions':report['input_projection']['discarded_velocity_coordinates'],
                      'projection_invariance_verified':True}))


if __name__ == '__main__':
    main()
