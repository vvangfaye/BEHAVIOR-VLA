"""Keep the two approved restoration evidence contracts explicit in aggregates."""
HISTORY = 'recorded_actions_before_start_native_grasp_recovery_v1'


def accepted_evidence_kind(row):
    if (row.get('expected_samples') != row.get('audited_samples')
            or row.get('all_required_video_sha_and_first_frames_passed') is not True):
        raise ValueError('Assignment lacks complete sample and video accounting')
    protocol = row.get('state_recovery_protocol')
    if protocol == HISTORY:
        required = ('original_scope_and_sample_ids_matched',
                    'independent_original_raw_and_prefix_hashes_matched',
                    'recovery_receipts_and_source_role_bindings_checked',
                    'all_history_setup_exclusions_checked_against_saved_evidence')
        if any(row.get(key) is not True for key in required):
            raise ValueError('History assignment lacks its original-scope artifact evidence')
        if (row.get('exact_original_grasp_anchor_bytes_verified') is not False
                or row.get('independent_all_raw_native_fields_matched') is not False
                or row.get('additional_grasp_physics_replayed_for_audit') is not False):
            raise ValueError('History assignment misstates the approved recovery evidence')
        ids = row.get('sample_ids', [])
        episodes = row.get('episode_keys', [])
        if (len(ids) != row['expected_samples'] or len(set(ids)) != len(ids)
                or len(episodes) != 4 or len({tuple(e) for e in episodes}) != 4):
            raise ValueError('History assignment does not cover its original four H5 files')
        return 'approved_physical_history'
    if protocol not in (None, 'recorded_native_fields_v1'):
        raise ValueError('Assignment uses an unrecognized restoration protocol')
    if (row.get('independent_all_raw_native_fields_matched') is not True
            or row.get('provided_source_role_bindings_independently_matched') is not True):
        raise ValueError('Native assignment lacks original state and role evidence')
    if row.get('setup_exclusions', {}).get('predicate_unobservable', 0):
        if row.get('all_role_unobservable_exclusions_rechecked_against_original_raw') is not True:
            raise ValueError('Native assignment lacks its exclusion evidence')
    return 'recorded_native_fields'
