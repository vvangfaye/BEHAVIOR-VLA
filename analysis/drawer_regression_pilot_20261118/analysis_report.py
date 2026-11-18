"""One prespecified drawer regression test on seed-disjoint observations.

Each row: seed, branch, signed_front_m (larger means further forward),
old_success, new_success. Success is whether the goal was achieved at any
point during the matched continuation, using the same horizon. Branch IDs
are unique within each source scene seed. Unavailable measurements or
outcomes are represented by null, never by a failure.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    raw = path.read_bytes()
    text = raw.decode()
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    return rows, hashlib.sha256(raw).hexdigest()


def classify(rows):
    by_seed = defaultdict(list)
    seen = set()
    missing_front = missing_outcome = 0
    for row in rows:
        seed, branch = row["seed"], row["branch"]
        if any(type(v) not in (int, str) or str(v) == "" for v in (seed, branch)):
            raise ValueError("seed and branch must be nonempty IDs")
        pair = str(seed), str(branch)
        if pair in seen:
            raise ValueError(f"duplicate seed/branch pair {pair}")
        seen.add(pair)
        front = row["signed_front_m"]
        if front is not None and (type(front) not in (int, float) or not math.isfinite(front)):
            raise ValueError(f"invalid signed_front_m at {pair}")
        missing_front += int(front is None)
        old, new = row["old_success"], row["new_success"]
        for outcome in (old, new):
            if outcome is not None and (type(outcome) not in (bool, int) or outcome not in (0, 1)):
                raise ValueError(f"invalid success flag at {pair}")
        missing_outcome += int(old is None or new is None)
        by_seed[str(seed)].append((front, old, new))
    counts = dict(rows=len(rows), source_seeds=len(by_seed),
                  rows_missing_front=missing_front, rows_missing_outcome=missing_outcome)
    return by_seed, counts


def seed_summaries(by_seed):
    # A single cluster gets a single location and a single response.
    # Use all measured locations, without looking at their outcomes.
    summaries = {}
    for seed, rows in by_seed.items():
        fronts = [f for f, _, _ in rows if f is not None]
        paired = [(int(o), int(n)) for f, o, n in rows if None not in (f, o, n)]
        gains = sum(n > o for o, n in paired)
        losses = sum(o > n for o, n in paired)
        summaries[seed] = dict(
            front_m=statistics.mean(fronts) if fronts else None,
            paired=len(paired), gains=gains, losses=losses,
            old_rate=statistics.mean(o for o, _ in paired) if paired else None,
            new_rate=statistics.mean(n for _, n in paired) if paired else None,
            old_minus_new=(losses - gains) / len(paired) if paired else None,
        )
    return summaries


def group_summary(group):
    return dict(
        seeds=len(group), paired_rows=sum(s["paired"] for s in group),
        old_only=sum(s["losses"] for s in group), new_only=sum(s["gains"] for s in group),
        seed_mean_old_success=statistics.mean(s["old_rate"] for s in group) if group else None,
        seed_mean_new_success=statistics.mean(s["new_rate"] for s in group) if group else None,
        seed_mean_old_minus_new=statistics.mean(s["old_minus_new"] for s in group) if group else None,
    )


def permutation_test(front, back):
    # Permute whole seed outcomes, preserving the observed group sizes.
    scores = [s["old_minus_new"] for s in front + back]
    n, m = len(front), len(back)
    total = sum(scores)
    observed = sum(scores[:n]) / n - sum(scores[n:]) / m
    def as_extreme(indices):
        selected = sum(scores[i] for i in indices)
        return selected / n - (total - selected) / m >= observed - 1e-12
    possibilities = math.comb(n + m, n)
    if possibilities <= 100_000:
        extreme = sum(as_extreme(ids) for ids in itertools.combinations(range(n + m), n))
        return dict(method="exact", allocations=possibilities,
                    observed_front_minus_back=observed, p_one_sided=extreme / possibilities)
    rng = random.Random(20261118)
    draws = 49_999
    extreme = sum(as_extreme(rng.sample(range(n + m), n)) for _ in range(draws))
    return dict(method="Monte Carlo with plus-one correction", possible_allocations=possibilities,
                draws=draws, random_seed=20261118,
                observed_front_minus_back=observed, p_one_sided=(extreme + 1) / (draws + 1))


def analyze(discovery_rows, holdout_rows=None):
    discovery, discovery_counts = classify(discovery_rows)
    coordinates = [s["front_m"] for s in seed_summaries(discovery).values() if s["front_m"] is not None]
    cutoff = statistics.median(coordinates) if coordinates else None
    report = dict(
        hypothesis="old_minus_new continuation success is greater further forward",
        endpoint="success_once over a common continuation horizon",
        statistic="mean per-seed old-minus-new success in front minus the same mean in back",
        group_definition="front iff mean measured signed_front_m within source seed exceeds discovery median",
        scope="Association in eligible handoff scenes; the contrast does not identify a causal effect of bowl position.",
        discovery=discovery_counts, discovery_geometric_seeds=len(coordinates), cutoff_m=cutoff,
    )
    if holdout_rows is None:
        report["status"] = "awaiting_holdout" if coordinates else "inconclusive_no_discovery_locations"
        return report
    holdout, counts = classify(holdout_rows)
    if set(discovery) & set(holdout):
        raise ValueError("discovery and holdout share source seed IDs")
    report["holdout"] = counts
    if cutoff is None:
        report["status"] = "inconclusive_no_discovery_locations"
        return report
    summaries = seed_summaries(holdout)
    eligible = [s for s in summaries.values() if s["old_minus_new"] is not None]
    front = [s for s in eligible if s["front_m"] > cutoff]
    back = [s for s in eligible if s["front_m"] <= cutoff]
    report.update(front=group_summary(front), back=group_summary(back),
                  holdout_eligible_seeds=len(eligible))
    if min(len(front), len(back)) == 0:
        report["status"] = "inconclusive_empty_group"
        return report
    test = permutation_test(front, back)
    report["permutation"] = test
    if any(f is not None and (o is None or n is None) for pairs in holdout.values() for f, o, n in pairs):
        report["status"] = "inconclusive_incomplete_pairs"
    elif math.comb(len(front) + len(back), len(front)) < 20:
        report["status"] = "inconclusive_insufficient_permutation_resolution"
    elif test["observed_front_minus_back"] > 0 and test["p_one_sided"] <= 0.05:
        report["status"] = "directional_association_at_prespecified_0.05_in_eligible_holdout"
    else:
        report["status"] = "inconclusive_no_directional_support_at_0.05"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    discovery, discovery_sha = read_rows(args.discovery)
    holdout, holdout_sha = read_rows(args.holdout) if args.holdout else (None, None)
    report = analyze(discovery, holdout)
    report.update(discovery_file_sha256=discovery_sha, holdout_file_sha256=holdout_sha)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
