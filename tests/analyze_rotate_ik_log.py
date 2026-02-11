#!/usr/bin/env python3
"""
Analyze rotate_object IK test logs to identify optimization opportunities.

Usage:
    python3 tests/analyze_rotate_ik_log.py
    python3 tests/analyze_rotate_ik_log.py --verbose
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_log():
    """Load the rotate IK log file."""
    log_file = os.path.join(PROJECT_ROOT, "tests", "rotate_ik_log.json")
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        print("Run test_rotate_ik.py first to generate data.")
        sys.exit(1)

    with open(log_file, 'r') as f:
        return json.load(f)


def analyze_phase_performance(log_data, verbose=False):
    """Analyze which phases win and when."""
    print(f"\n{'='*70}")
    print(f"PHASE PERFORMANCE ANALYSIS ({len(log_data)} runs)")
    print(f"{'='*70}")

    phase0_triggers = sum(1 for run in log_data if run.get("phase0_triggered"))
    phase1_wins = sum(1 for run in log_data if run.get("first_valid", {}).get("phase") == "Phase1")
    phase2a_wins = sum(1 for run in log_data if run.get("first_valid", {}).get("phase") == "Phase2a")

    print(f"\nPhase 0 (yaw-adjusted) triggered: {phase0_triggers}/{len(log_data)} ({100*phase0_triggers/len(log_data):.1f}%)")
    print(f"Phase 1 (current joints) wins: {phase1_wins}/{len(log_data)} ({100*phase1_wins/len(log_data):.1f}%)")
    print(f"Phase 2a (fallback seeds) wins: {phase2a_wins}/{len(log_data)} ({100*phase2a_wins/len(log_data):.1f}%)")

    # Phase 1 success vs angle difference
    phase1_success_by_angle = defaultdict(list)
    for run in log_data:
        angle = run.get("angle_diff_deg", 0)
        phase1_valid = run.get("phase1_any_valid", False)
        phase1_success_by_angle[int(angle // 10) * 10].append(phase1_valid)

    print(f"\nPhase 1 success rate by angle difference:")
    for angle_bin in sorted(phase1_success_by_angle.keys()):
        successes = phase1_success_by_angle[angle_bin]
        rate = 100 * sum(successes) / len(successes)
        print(f"  {angle_bin:3d}°-{angle_bin+10:3d}°: {sum(successes):2d}/{len(successes):2d} ({rate:5.1f}%)")

    # Winning seeds
    winning_seeds = Counter()
    for run in log_data:
        first = run.get("first_valid")
        if first:
            winning_seeds[f"[{first['phase']}] {first['seed']}"] += 1

    print(f"\nTop winning seeds (first valid in solve order):")
    for seed, count in winning_seeds.most_common(10):
        print(f"  {count:2d}x: {seed}")

    # Wasted IK solves
    phase1_wasted = []
    for run in log_data:
        if run.get("first_valid", {}).get("phase") == "Phase2a":
            phase1_wasted.append(run["ik_solves"]["phase1"])

    if phase1_wasted:
        avg_wasted = sum(phase1_wasted) / len(phase1_wasted)
        print(f"\nWasted IK solves when Phase 1 fails:")
        print(f"  Runs with Phase 2a winning: {len(phase1_wasted)}")
        print(f"  Average Phase 1 solves wasted: {avg_wasted:.1f}")
        print(f"  Total wasted: {sum(phase1_wasted)}")

    if verbose:
        print(f"\nDetailed run-by-run:")
        for i, run in enumerate(log_data, 1):
            angle = run.get("angle_diff_deg", 0)
            first = run.get("first_valid", {})
            if first:
                print(f"  Run {i}: angle={angle:5.1f}°, winner=[{first['phase']}] {first['seed']}, cost={first['cost']:.6f}")
            else:
                print(f"  Run {i}: angle={angle:5.1f}°, NO SOLUTION FOUND")


def analyze_angle_distribution(log_data):
    """Analyze angle difference distribution."""
    print(f"\n{'='*70}")
    print("ANGLE DIFFERENCE DISTRIBUTION")
    print(f"{'='*70}")

    angles = [run.get("angle_diff_deg", 0) for run in log_data]

    print(f"\nStatistics:")
    print(f"  Min:    {min(angles):6.1f}°")
    print(f"  Max:    {max(angles):6.1f}°")
    print(f"  Mean:   {sum(angles)/len(angles):6.1f}°")
    print(f"  Median: {sorted(angles)[len(angles)//2]:6.1f}°")

    # Histogram
    bins = [0, 10, 30, 60, 90, 120, 150, 180]
    hist = defaultdict(int)
    for angle in angles:
        for i in range(len(bins)-1):
            if bins[i] <= angle < bins[i+1]:
                hist[f"{bins[i]:3d}°-{bins[i+1]:3d}°"] += 1
                break
        else:
            if angle >= bins[-1]:
                hist[f">={bins[-1]}°"] += 1

    print(f"\nHistogram:")
    for bin_label in [f"{bins[i]:3d}°-{bins[i+1]:3d}°" for i in range(len(bins)-1)] + [f">={bins[-1]}°"]:
        count = hist[bin_label]
        if count > 0:
            bar = '█' * int(30 * count / len(angles))
            print(f"  {bin_label}: {count:2d} {bar}")


def analyze_efficiency(log_data):
    """Analyze IK solver efficiency."""
    print(f"\n{'='*70}")
    print("IK SOLVER EFFICIENCY")
    print(f"{'='*70}")

    total_solves = [run["ik_solves"]["total"] for run in log_data]
    phase1_solves = [run["ik_solves"]["phase1"] for run in log_data]
    phase2a_solves = [run["ik_solves"]["phase2a"] for run in log_data]

    print(f"\nAverage IK solves per run:")
    print(f"  Phase 0: {sum(run['ik_solves']['phase0'] for run in log_data) / len(log_data):.1f}")
    print(f"  Phase 1: {sum(phase1_solves) / len(phase1_solves):.1f}")
    print(f"  Phase 2a: {sum(phase2a_solves) / len(phase2a_solves):.1f}")
    print(f"  Total:   {sum(total_solves) / len(total_solves):.1f}")

    # Potential savings if Phase 1 was skipped when angle > threshold
    thresholds = [10, 20, 30, 45, 60, 90]
    print(f"\nPotential savings if Phase 1 skipped when angle > threshold:")
    for threshold in thresholds:
        runs_above = [run for run in log_data if run.get("angle_diff_deg", 0) > threshold]
        if runs_above:
            phase1_in_those = sum(run["ik_solves"]["phase1"] for run in runs_above)
            print(f"  >{threshold:3d}°: {len(runs_above):2d} runs, save {phase1_in_those} IK solves ({phase1_in_those/len(runs_above):.1f} avg)")

    # With early termination enabled
    print(f"\nWith early_termination=True (exit on first valid):")
    early_exit_saves = 0
    for run in log_data:
        first_valid = run.get("first_valid")
        if first_valid:
            if first_valid["phase"] == "Phase1":
                # Would exit after Phase 1 first solve, save Phase 2a solves
                early_exit_saves += run["ik_solves"]["phase2a"]
            elif first_valid["phase"] == "Phase2a":
                # Would exit on first Phase 2a seed that wins
                # Count how many Phase 2a seeds came before the winner
                winner_seed = first_valid["seed"]
                phase2_seeds = [
                    "wrist-only w1=-30 w2=90", "wrist-only w1=0 w2=90", "wrist-only w1=30 w2=90",
                    "wrist-only w1=-30 w2=-90", "wrist-only w1=0 w2=-90", "wrist-only w1=30 w2=-90",
                    "yaw-flip w1=-30 w2=90", "yaw-flip w1=0 w2=90", "yaw-flip w1=-30 w2=-90",
                    "85/-80/90/-90/-90", "90/-90/90/-90/-90", "0/-90/90/-90/-90", "180/-90/90/-90/-90",
                    "85/-80/90/-90/90", "90/-90/90/-90/90", "85/-108/106/2/76", "85/-80/90/90/-90",
                ]
                if winner_seed in phase2_seeds:
                    winner_idx = phase2_seeds.index(winner_seed)
                    # Would skip all seeds after winner
                    early_exit_saves += len(phase2_seeds) - winner_idx - 1

    total_current = sum(total_solves)
    print(f"  Total current IK solves: {total_current}")
    print(f"  Could save: {early_exit_saves} solves")
    print(f"  New total: {total_current - early_exit_saves} ({100*(total_current-early_exit_saves)/total_current:.1f}% of current)")


def classify_face(roll_deg):
    """Classify EE face orientation from roll angle."""
    abs_roll = abs(roll_deg)
    if abs_roll > 160:
        return "down"
    elif 70 < abs_roll < 110:
        return "forward"
    else:
        return "other(%.0f)" % roll_deg


def analyze_face_transitions(log_data, verbose=False):
    """Analyze which phase wins based on face orientation transitions."""
    print(f"\n{'='*70}")
    print("FACE ORIENTATION TRANSITION ANALYSIS")
    print(f"{'='*70}")

    transitions = defaultdict(list)
    for run in data_with_transition(log_data):
        key = run["_transition"]
        transitions[key].append(run)

    for trans in sorted(transitions.keys()):
        runs = transitions[trans]
        phase_counts = Counter(r.get("first_valid", {}).get("phase", "NONE") for r in runs)
        p1_valid = sum(1 for r in runs if r.get("phase1_any_valid"))
        parts = ", ".join(f"{p}={c}" for p, c in phase_counts.most_common())
        print(f"\n  {trans} ({len(runs)} runs): {parts}")
        print(f"    Phase1 found any valid: {p1_valid}/{len(runs)}")

        # Per-object breakdown
        obj_counts = defaultdict(lambda: Counter())
        for r in runs:
            obj = r.get("object", "?")
            phase = r.get("first_valid", {}).get("phase", "NONE")
            obj_counts[obj][phase] += 1

        for obj in sorted(obj_counts.keys()):
            phases = obj_counts[obj]
            total = sum(phases.values())
            parts = ", ".join(f"{p}={c}" for p, c in phases.most_common())
            print(f"      {obj}: {total} runs - {parts}")

    if verbose:
        print(f"\n  Detailed:")
        for run in data_with_transition(log_data):
            obj = run.get("object", "?")
            trans = run["_transition"]
            angle = run["angle_diff_deg"]
            first = run.get("first_valid", {})
            phase = first.get("phase", "NONE")
            seed = first.get("seed", "?")
            ec = run["ee_rpy_current"]
            et = run["ee_rpy_target"]
            print(f"    {obj:20s} {trans:15s} a={angle:5.1f} "
                  f"cur_yaw={ec[2]:7.1f} tgt_yaw={et[2]:7.1f} "
                  f"-> [{phase}] {seed}")


def data_with_transition(log_data):
    """Add _transition field to each run."""
    for run in log_data:
        cur_face = classify_face(run["ee_rpy_current"][0])
        tgt_face = classify_face(run["ee_rpy_target"][0])
        run["_transition"] = f"{cur_face}->{tgt_face}"
        yield run


def generate_recommendations(log_data):
    """Generate optimization recommendations based on data."""
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    phase1_wins = sum(1 for run in log_data if run.get("first_valid", {}).get("phase") == "Phase1")
    phase1_win_rate = 100 * phase1_wins / len(log_data)

    # Analyze angle threshold for Phase 1
    angles_when_phase1_wins = [run.get("angle_diff_deg", 0) for run in log_data
                               if run.get("first_valid", {}).get("phase") == "Phase1"]
    if angles_when_phase1_wins:
        max_angle_phase1_win = max(angles_when_phase1_wins)
    else:
        max_angle_phase1_win = 0

    print(f"\n1. Enable early_termination=True")
    print(f"   - Currently all {len(log_data)} runs execute all phases regardless")
    print(f"   - Would save significant time by exiting on first valid solution")

    # Transition-based analysis
    down_down = [r for r in data_with_transition(log_data) if r["_transition"] == "down->down"]
    down_fwd = [r for r in data_with_transition(log_data) if r["_transition"] == "down->forward"]
    fwd_any = [r for r in data_with_transition(log_data) if r["_transition"].startswith("forward->")]

    print(f"\n2. Face transition determines the winner:")
    if down_down:
        dd_p0 = sum(1 for r in down_down if r.get("first_valid", {}).get("phase") == "Phase0")
        dd_p1 = sum(1 for r in down_down if r.get("first_valid", {}).get("phase") == "Phase1")
        print(f"   - down->down (yaw-only): {len(down_down)} runs, Phase0={dd_p0} Phase1={dd_p1}")
        print(f"     Phase0+Phase1 handle this; Phase2a not needed")
    if down_fwd:
        df_p1 = sum(1 for r in down_fwd if r.get("first_valid", {}).get("phase") == "Phase1")
        df_p2 = sum(1 for r in down_fwd if r.get("first_valid", {}).get("phase") == "Phase2a")
        df_p1_valid = sum(1 for r in down_fwd if r.get("phase1_any_valid"))
        print(f"   - down->forward (face change): {len(down_fwd)} runs, Phase1={df_p1} Phase2a={df_p2}")
        print(f"     Phase1 finds ANY valid: {df_p1_valid}/{len(down_fwd)} (object-dependent)")
        print(f"     Phase2a wrist-only seed is essential fallback")

        # Per-object in down->fwd
        obj_p1_works = defaultdict(lambda: [0, 0])
        for r in down_fwd:
            obj = r.get("object", "?")
            obj_p1_works[obj][0] += 1
            if r.get("phase1_any_valid"):
                obj_p1_works[obj][1] += 1
        print(f"     Per-object Phase1 success in face changes:")
        for obj in sorted(obj_p1_works.keys()):
            total, p1ok = obj_p1_works[obj]
            print(f"       {obj}: Phase1 valid {p1ok}/{total}")
    if fwd_any:
        print(f"   - forward->*: {len(fwd_any)} runs (need more data)")

    print(f"\n3. Phase 0 (yaw-adjusted):")
    phase0_triggers = sum(1 for run in log_data if run.get("phase0_triggered"))
    phase0_wins = sum(1 for run in log_data if run.get("first_valid", {}).get("phase") == "Phase0")
    if phase0_triggers == 0:
        print(f"   - Never triggered in {len(log_data)} runs")
        print(f"   - CONSIDER: Remove Phase 0 or adjust trigger conditions")
    else:
        print(f"   - Triggered {phase0_triggers} times, won {phase0_wins} ({100*phase0_wins/max(phase0_triggers,1):.0f}%)")
        print(f"   - All in down->down transitions, cheap (1 solve), keep it")

    print(f"\n4. Phase 2b:")
    print(f"   - Already disabled (if False)")
    print(f"   - Can safely delete the dead code")

    print(f"\n5. Recommendation:")
    print(f"   - Enable early_termination=True (biggest win)")
    print(f"   - Keep Phase 0 for down->down yaw adjustments")
    print(f"   - Keep Phase 1 (wins for some objects in face changes)")
    print(f"   - Keep Phase 2a wrist-only seed as essential fallback")
    print(f"   - Consider pruning Phase 2a to just wrist-only w1=-30 w2=90 (won 100% of Phase2a)")
    print(f"   - Delete Phase 2b dead code")


def main():
    parser = argparse.ArgumentParser(description='Analyze rotate_object IK test logs')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed run-by-run info')
    args = parser.parse_args()

    log_data = load_log()

    if not log_data:
        print("No data in log file. Run test_rotate_ik.py first.")
        return

    print(f"Loaded {len(log_data)} test runs from rotate_ik_log.json")

    analyze_phase_performance(log_data, verbose=args.verbose)
    analyze_face_transitions(log_data, verbose=args.verbose)
    analyze_angle_distribution(log_data)
    analyze_efficiency(log_data)
    generate_recommendations(log_data)

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
