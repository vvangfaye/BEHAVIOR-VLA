import copy
import unittest

from score_finetune_development import compare, validate


def bundle(pairs, successes, *, checkpoint="b", q_failure=.2):
    return dict(kind="artifact_accepted_finetune_development_results", policy_seed=42,
        checkpoint_manifest_sha256=checkpoint*64, protocol_sha256="c"*64, acceptance_manifest_sha256="d"*64,
        records=[dict(task=t, instance_id=i, policy_seed=42, steps=100,
            model_success=(t,i) in successes, q_score=1 if (t,i) in successes else q_failure,
            result_sha256="e"*64, artifact_accepted=True, policy_seed_acknowledged=True) for t,i in pairs])


class PairedDevelopmentTests(unittest.TestCase):
    def test_macro_task_metric_is_not_micro_instance_metric(self):
        pairs=[("many",i) for i in range(3)]+[("single",0)]
        old=bundle(pairs,{("many",0),("many",1),("many",2)},checkpoint="a")
        new=bundle(pairs,{("single",0)},checkpoint="b")
        result=compare(old,new,pairs,completed_updates=5000)
        self.assertAlmostEqual(result["baseline_macro_task_success"],.5)
        self.assertAlmostEqual(result["candidate_macro_task_success"],.5)
        self.assertFalse(result["may_continue_past5000"])

    def test_quick20_and_wrong_seed_cannot_grant_continuation(self):
        pairs=[("t"+str(i),0) for i in range(20)]
        old=bundle(pairs,set(),checkpoint="a");new=bundle(pairs,set(pairs),checkpoint="b")
        self.assertEqual(compare(old,new,pairs,completed_updates=5000,full_development=False)["decision"],"technical_check_only")
        wrong=copy.deepcopy(new);wrong["policy_seed"]=43
        with self.assertRaises(ValueError):compare(old,wrong,pairs,completed_updates=5000)

    def test_missing_duplicate_or_unverified_evidence_is_rejected(self):
        pairs=[("t",0),("t",1)];good=bundle(pairs,set())
        for mode in ("missing","duplicate","seed","infrastructure"):
            bad=copy.deepcopy(good)
            if mode=="missing":bad["records"].pop()
            if mode=="duplicate":bad["records"][1]=bad["records"][0]
            if mode=="seed":bad["records"][0]["policy_seed_acknowledged"]=False
            if mode=="infrastructure":bad["records"][0]["artifact_accepted"]=False
            with self.assertRaises(ValueError):validate(bad,pairs,42)

    def test_full_scope_improvement_and_regression_guard(self):
        pairs=[("task"+str(i),0) for i in range(102)]
        old=bundle(pairs,set(pairs[:8]),checkpoint="a")
        improved=bundle(pairs,set(pairs[:9]),checkpoint="b")
        result=compare(old,improved,pairs,completed_updates=5000)
        self.assertTrue(result["may_continue_past5000"])
        final=compare(old,improved,pairs,completed_updates=20000)
        self.assertFalse(final["may_continue_past5000"])
        self.assertEqual(final["decision"],"budget_complete_select_best")
        with self.assertRaises(ValueError):compare(old,improved,pairs,completed_updates=20001)
        worse=bundle(pairs,set(pairs[:3]),checkpoint="b",q_failure=.1)
        result=compare(old,worse,pairs,completed_updates=5000)
        self.assertEqual(result["net_additional_failed_pairs"],5)
        self.assertEqual(result["decision"],"review_required_regression")


if __name__=="__main__":unittest.main()
