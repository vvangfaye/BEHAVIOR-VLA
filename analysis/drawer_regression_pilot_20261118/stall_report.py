"""Prospective secondary test at one new-policy drawer landmark per scene."""
import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def first_present(row, *keys):
    for key in keys:
        if key in row:
            return row[key]
    return None


def classify(row, horizon=520):
    """Recompute the geometric flag from the four 10-action boundaries."""
    first = first_present(row, "first_in_step", "handoff_step")
    step = row["landmark_step"]
    window = first_present(row, "window", "metrics_last4") or []
    eligible, flagged = row["eligible"], row["flagged"]
    if type(eligible) is not bool or flagged is not None and type(flagged) is not bool:
        raise ValueError("eligible/flagged must be boolean; unobserved flag may be null")
    if eligible and not window:
        raise ValueError("eligible landmark lacks four boundary observations")
    if window:
        if first is None or step != first + 80 or step % 10 or step > horizon:
            raise ValueError("landmark must be the first boundary + 80 actions")
        if [m["step"] for m in window] != [step - 30, step - 20, step - 10, step]:
            raise ValueError("landmark must contain all four consecutive boundaries")
        if any(type(m["bowl_in"]) is not bool or type(m["drawer_closed"]) is not bool
               or not math.isfinite(m["drawer_open_m"]) for m in window):
            raise ValueError("invalid landmark measurements")
        success = first_present(row, "root_first_success_step", "root_new_first_success_step")
        if success is not None and success <= step and eligible:
            raise ValueError("landmark after new root had already succeeded")
        last = window[-1]
        geom_eligible = (step < horizon and last["bowl_in"] and not last["drawer_closed"]
                         and (success is None or success > step))
        if eligible != geom_eligible:
            raise ValueError("native open eligibility disagrees with landmark")
        net_close = window[0]["drawer_open_m"] - last["drawer_open_m"]
        computed = (all(m["bowl_in"] for m in window) and last["drawer_open_m"] > 0.04 and net_close < 0.01)
        if eligible and flagged != computed:
            raise ValueError("saved flag disagrees with four frozen measurements")
    if not eligible:
        if flagged not in (False, None) or any(row.get(k) is not None for k in ("old_success", "new_success")):
            raise ValueError("ineligible scene has a positive flag or continued outcomes")
        reason = row.get("ineligible_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("ineligible root lacks a censoring reason")
        return "ineligible", reason
    if any(row.get(k) is not None and type(row[k]) is not bool for k in ("old_success", "new_success")):
        raise ValueError("continuation outcomes must be boolean or missing")
    return ("flagged" if flagged else "other"), None


def binomial_tail(p, n, lo, hi):
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(lo, hi + 1))


def monotone_root(fn, target):
    lo, hi = 0.0, 1.0
    for _ in range(54):
        mid = (lo + hi) / 2
        if fn(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def binomial_ci(k, n, alpha):
    # Exact two-sided Clopper-Pearson interval, including endpoint counts.
    lo = 0.0 if k == 0 else monotone_root(lambda p: binomial_tail(p, n, k, n), alpha / 2)
    hi = 1.0 if k == n else monotone_root(lambda p: 1 - binomial_tail(p, n, 0, k), 1 - alpha / 2)
    return lo, hi


def arm_summary(rows):
    complete = [r for r in rows if None not in (r.get("old_success"), r.get("new_success"))]
    counts = Counter((r["old_success"], r["new_success"]) for r in complete)
    n, loss, gain = len(complete), counts[True, False], counts[False, True]
    result = dict(eligible=len(rows), paired=n, missing_pairs=len(rows) - n,
                  old_only=loss, new_only=gain, both_success=counts[True, True], neither=counts[False, False],
                  old_minus_new=(loss - gain) / n if n else None)
    if n:
        # Probability difference = p(old only) - p(new only); combine simultaneous bounds.
        a, b = binomial_ci(loss, n, 0.025), binomial_ci(gain, n, 0.025)
        result["old_minus_new_95pct_conservative_interval"] = [a[0] - b[1], a[1] - b[0]]
    return result


def analyze(rows, expected_seeds):
    if len(set(expected_seeds)) != len(expected_seeds) or len(expected_seeds) > 20:
        raise ValueError("expected distinct source seeds for a bounded exact test")
    groups = defaultdict(list)
    reasons = Counter()
    seen = set()
    for row in rows:
        seed = row["seed"]
        if row.get("split", "formal_holdout") != "formal_holdout":
            raise ValueError("Only prespecified formal holdout records are eligible")
        if seed not in expected_seeds or seed in seen:
            raise ValueError(f"unexpected or duplicate seed {seed}")
        seen.add(seed)
        group, reason = classify(row)
        groups[group].append(row)
        if reason:
            reasons[reason] += 1
    flagged, other = groups["flagged"], groups["other"]
    all_open = flagged + other
    result = dict(
        scope="Prespecified secondary holdout screen on new-policy roots, conditional on reaching an open handoff.",
        landmark="First new-root open bowl-in boundary plus exactly 80 actions; eligibility uses native predicates.",
        flag="At landmark bowl_in at t-30,t-20,t-10,t, drawer_open_m(t)>0.04 and opening(t-30)-opening(t)<0.01.",
        primary_contrast="(old minus new continuation success) in flagged minus other eligible roots at the same landmark.",
        endpoint="success_once before the original 520-action horizon, one matched policy pair per source seed",
        expected_seeds=list(expected_seeds), roots_observed=len(rows),
        missing_roots=sorted(set(expected_seeds) - seen), ineligible_reasons=dict(reasons),
        censored_after_already_succeeded=sum(
            r.get("root_first_success_step") is not None and
            (r.get("first_in_step") is None or r["root_first_success_step"] <= r["first_in_step"] + 80)
            for r in groups["ineligible"]
        ),
        eligible_fraction=len(all_open) / len(rows) if rows else None,
        flagged_fraction_of_all_observed=len(flagged) / len(rows) if rows else None,
        any_still_open=arm_summary(all_open), flagged=arm_summary(flagged), other_still_open=arm_summary(other),
        uncertainty="Exact conditional test under exchangeability of seed-level differences across the two groups. Descriptive subgroup intervals account for both discordant outcomes.",
        early_completions="Roots that already succeeded do not enter the continuation comparison; the flag is not a whole-task policy effect.",
    )
    f, o = result["flagged"], result["other_still_open"]
    if result["missing_roots"] or result["any_still_open"]["missing_pairs"]:
        result["status"] = "inconclusive_incomplete_formal_screen"
    elif not f["paired"] or not o["paired"]:
        result["status"] = "inconclusive_no_flagged_or_no_landmark_controls"
    else:
        scores = [int(r["old_success"]) - int(r["new_success"]) for r in all_open]
        n, k = len(scores), len(flagged)
        # With fixed group counts the difference of means is strictly increasing
        # in the flagged sum, permitting an exact test on integers including ties.
        actual_sum = sum(scores[:k])
        extreme = sum(sum(scores[i] for i in ids) >= actual_sum
                      for ids in itertools.combinations(range(n), k))
        choices = math.comb(n, k)
        p = extreme / choices
        effect = f["old_minus_new"] - o["old_minus_new"]
        result["exact_one_sided_test"] = dict(allocations=choices, contrast=effect, p=p)
        if choices < 20:
            result["status"] = "inconclusive_insufficient_possible_allocations"
        elif effect > 0 and p <= 0.05:
            result["status"] = "directional_secondary_association_in_eligible_holdout"
        else:
            result["status"] = "inconclusive_no_directional_evidence_at_0.05"
    return result


def parse_seeds(arg):
    if ":" in arg:
        start, end = (int(x) for x in arg.split(":"))
        return list(range(start, end + 1))
    return [int(x) for x in arg.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--seeds", required=True, help="comma-separated IDs or inclusive start:end")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = parse_seeds(args.seeds)
    rows = [json.loads(f.read_text()) for f in sorted(args.input_dir.glob("paired_*.json"))]
    result = analyze(rows, expected)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "roots_observed", "flagged_fraction_of_all_observed")}))


if __name__ == "__main__":
    main()
