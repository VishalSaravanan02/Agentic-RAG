# =============================================================================
# render_failure_table.py — Render the Phase 7 tables from failure_analysis.json
#
# Every figure in the qualitative failure-mode section is produced here, so the
# tables regenerate from committed code rather than being transcribed by hand.
#
# Proportions are reported with 95% Wilson score intervals. No finite population
# correction is applied: sampling 50 of 395 failures would narrow each interval
# by about 6%, so omitting it is slightly conservative. The intervals are wide
# and are printed deliberately: at this sample size the categories cannot be
# ranked against one another, and the tables should not be read as though they
# can.
#
# An earlier version re-centred the Wilson half-width on the raw proportion
# k/n. That discards the interval's asymmetry and, for small counts, roughly
# halves the upper bound (0 of 50 was reported as at most 3.3%; the Wilson
# upper bound is 7.1%). Corrected before the dissertation was finalised.
#
# Usage:
#   python scripts/render_failure_table.py                # print, save JSON and SVG
#   python scripts/render_failure_table.py --no-chart     # skip the SVG
#   python scripts/render_failure_table.py --no-save      # skip the JSON
# =============================================================================

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import RESULTS_DIR

ANALYSIS_PATH = os.path.join(RESULTS_DIR, "failure_analysis.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "failure_modes_eval.json")
CHART_PATH = os.path.join(RESULTS_DIR, "failure_modes_eval.svg")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def interval(k: int, n: int, N: int) -> tuple:
    """95% Wilson score interval for k of n.

    N (the population size) is accepted so that callers can report the implied
    range of cases in the population, but it does not alter the interval: no
    finite population correction is applied (see header).
    """
    lo, hi = wilson(k, n)
    return max(0.0, lo), min(1.0, hi)


def load() -> dict:
    if not os.path.exists(ANALYSIS_PATH):
        raise FileNotFoundError(f"{ANALYSIS_PATH} not found.")
    with open(ANALYSIS_PATH, "r") as f:
        return json.load(f)


def render_text(d: dict) -> None:
    cases = d["cases"]
    n = len(cases)
    N = d["meta"]["population"]["total_failures"]

    print("=" * 78)
    print(f"PHASE 7 FAILURE-MODE ANALYSIS   n = {n} of {N} failures "
          f"({n / N:.1%}), stratified by question type, seed {d['meta']['sample_seed']}")
    print("=" * 78)

    print("\nPRIMARY CATEGORY")
    print(f"  {'category':<26}{'n':>4}{'share':>9}   {'95% CI':>16}   implied in population")
    for cat, k in Counter(c["primary"] for c in cases).most_common():
        lo, hi = interval(k, n, N)
        print(f"  {cat:<26}{k:>4}{k / n:>8.0%}   [{lo:>5.1%}, {hi:>5.1%}]   [{lo * N:>5.0f}, {hi * N:>5.0f}]")

    empty = [c for c in d["meta"]["categories"] if c not in {x["primary"] for x in cases}]
    for cat in empty:
        lo, hi = interval(0, n, N)
        print(f"  {cat:<26}{0:>4}{0:>8.0%}   [{lo:>5.1%}, {hi:>5.1%}]   [{lo * N:>5.0f}, {hi * N:>5.0f}]")

    sec = Counter(s for c in cases for s in c["secondary"])
    if sec:
        print("\nSECONDARY CATEGORY (cases carrying more than one)")
        for cat, k in sec.most_common():
            print(f"  {cat:<26}{k:>4}")
        multi = [c["n"] for c in cases if c["secondary"]]
        print(f"  cases: {multi}")

    sub = Counter(c.get("unanswerable_subtype") for c in cases if c["primary"] == "unanswerable")
    if sub:
        print("\nUNANSWERABLE SUBTYPE")
        for k, v in sub.most_common():
            print(f"  {k:<26}{v:>4}")

    print("\nATTRIBUTES (recorded independently of category)")
    for attr in d["meta"]["attributes"]:
        ids = [c["n"] for c in cases if c["attributes"].get(attr)]
        lo, hi = interval(len(ids), n, N)
        print(f"  {attr:<26}{len(ids):>4}{len(ids) / n:>8.0%}   [{lo:>5.1%}, {hi:>5.1%}]   cases {ids}")

    nofault = [c["n"] for c in cases
               if c["primary"] == "unanswerable" or c["attributes"].get("second_valid_answer")]
    lo, hi = interval(len(nofault), n, N)
    print(f"\nNO SYSTEM FAULT (unanswerable or second valid answer)")
    print(f"  {len(nofault)}/{n} = {len(nofault) / n:.0%}   [{lo:.1%}, {hi:.1%}]   cases {nofault}")

    print("\nCATEGORY BY EVIDENCE COVERAGE")
    t = defaultdict(Counter)
    for c in cases:
        t[c["primary"]][c["coverage"]] += 1
    for cat in sorted(t):
        print(f"  {cat:<26}{dict(sorted(t[cat].items()))}")

    print("\nCATEGORY BY STOP CONDITION")
    t = defaultdict(Counter)
    for c in cases:
        t[c["primary"]][c["stop_condition"]] += 1
    for cat in sorted(t):
        print(f"  {cat:<26}{dict(t[cat])}")

    print("\nCATEGORY BY QUESTION TYPE")
    t = defaultdict(Counter)
    for c in cases:
        t[c["type"]][c["primary"]] += 1
    for qt in sorted(t):
        print(f"  {qt:<26}{dict(t[qt])}")

    print("\nNOTE  Intervals are wide at n = 50. Categories whose intervals overlap "
          "cannot be\n      ranked against one another; the counts establish that each mode "
          "occurs and\n      is not negligible, not which is largest.")
    print("=" * 78)


def render_chart(d: dict, path: str) -> str:
    """
    Write a horizontal bar chart of the primary failure modes as SVG.

    Bars show the share of the sample; whiskers show the 95% interval. The
    chart exists to make the overlap between intervals visible at a glance,
    since that overlap is what prevents the modes being ranked against one
    another at this sample size. Written as raw SVG so that no plotting
    dependency is added to an environment frozen for the experiments.
    """
    cases = d["cases"]
    n = len(cases)
    N = d["meta"]["population"]["total_failures"]

    counts = Counter(c["primary"] for c in cases)
    rows = [(cat, counts.get(cat, 0)) for cat in d["meta"]["categories"]]
    rows.sort(key=lambda r: -r[1])

    W, LEFT, RIGHT, TOP, ROW = 860, 210, 70, 92, 42
    H = TOP + ROW * len(rows) + 78
    span = W - LEFT - RIGHT
    xmax = 0.45
    x = lambda v: LEFT + (v / xmax) * span

    INK, MUTE, BAR, RULE = "#1a1a1a", "#666666", "#4a6fa5", "#d8d8d8"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="24" y="34" font-size="17" font-weight="600" fill="{INK}">'
           f'Primary failure mode, {n} sampled Main System failures</text>',
           f'<text x="24" y="56" font-size="12.5" fill="{MUTE}">'
           f'Bars show share of sample; whiskers show 95% interval '
           f'(Wilson score, n={n} of N={N}).</text>']

    for pct in range(0, int(xmax * 100) + 1, 10):
        gx = x(pct / 100)
        out.append(f'<line x1="{gx:.1f}" y1="{TOP - 12}" x2="{gx:.1f}" y2="{TOP + ROW * len(rows) - 12}" '
                   f'stroke="{RULE}" stroke-width="1"/>')
        out.append(f'<text x="{gx:.1f}" y="{TOP + ROW * len(rows) + 8}" font-size="11.5" '
                   f'fill="{MUTE}" text-anchor="middle">{pct}%</text>')

    for i, (cat, k) in enumerate(rows):
        cy = TOP + ROW * i + 6
        lo, hi = interval(k, n, N)
        p_hat = k / n
        out.append(f'<text x="{LEFT - 14}" y="{cy + 4}" font-size="13" fill="{INK}" '
                   f'text-anchor="end">{cat}</text>')
        if k:
            out.append(f'<rect x="{LEFT}" y="{cy - 10}" width="{x(p_hat) - LEFT:.1f}" height="20" '
                       f'fill="{BAR}" rx="2"/>')
        out.append(f'<line x1="{x(lo):.1f}" y1="{cy}" x2="{x(hi):.1f}" y2="{cy}" '
                   f'stroke="{INK}" stroke-width="1.4"/>')
        for e in (lo, hi):
            out.append(f'<line x1="{x(e):.1f}" y1="{cy - 6}" x2="{x(e):.1f}" y2="{cy + 6}" '
                       f'stroke="{INK}" stroke-width="1.4"/>')
        out.append(f'<text x="{x(hi) + 10:.1f}" y="{cy + 4}" font-size="12.5" fill="{MUTE}">'
                   f'{k}  ({p_hat:.0%})</text>')

    out.append(f'<text x="24" y="{H - 34}" font-size="12" fill="{MUTE}">'
               f'Intervals overlap substantially: the modes cannot be ranked against one '
               f'another at this sample size.</text>')
    out.append(f'<text x="24" y="{H - 15}" font-size="12" fill="{MUTE}">'
               f'The counts establish that each mode occurs and is not negligible, not which is largest.</text>')
    out.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out))
    return path


def save(d: dict, path: str) -> str:
    """Write the computed counts to RESULTS_DIR, overwriting any previous run."""
    cases = d["cases"]
    n = len(cases)
    N = d["meta"]["population"]["total_failures"]

    def block(counter):
        out = {}
        for k, v in counter.items():
            lo, hi = interval(v, n, N)
            out[k] = {"n": v, "share": round(v / n, 4),
                      "ci_95": [round(lo, 4), round(hi, 4)],
                      "implied_in_population": [round(lo * N), round(hi * N)]}
        return out

    primary = Counter(c["primary"] for c in cases)
    for cat in d["meta"]["categories"]:
        primary.setdefault(cat, 0)

    attributes = {}
    for attr in d["meta"]["attributes"]:
        ids = [c["n"] for c in cases if c["attributes"].get(attr)]
        lo, hi = interval(len(ids), n, N)
        attributes[attr] = {"n": len(ids), "share": round(len(ids) / n, 4),
                            "ci_95": [round(lo, 4), round(hi, 4)], "cases": ids}

    nofault = [c["n"] for c in cases
               if c["primary"] == "unanswerable" or c["attributes"].get("second_valid_answer")]
    lo, hi = interval(len(nofault), n, N)

    by = lambda key: {k: dict(v) for k, v in
                      ((a, Counter(c[key] for c in cases if c["primary"] == a))
                       for a in sorted({c["primary"] for c in cases}))}

    payload = {
        "generated_at": datetime.now().isoformat(),
        "split": d["meta"]["split"],
        "n_sampled": n,
        "n_population": N,
        "sample_seed": d["meta"]["sample_seed"],
        "interval_method": "Wilson score, 95%, no finite population correction "
                           "(conservative; the correction would narrow intervals by about 6%)",
        "caveat": "Intervals are wide at this sample size. Categories whose intervals "
                  "overlap cannot be ranked against one another; the counts establish "
                  "that each mode occurs and is not negligible, not which is largest.",
        "primary_category": block(primary),
        "secondary_category": dict(Counter(s for c in cases for s in c["secondary"])),
        "multi_label_cases": [c["n"] for c in cases if c["secondary"]],
        "unanswerable_subtype": dict(Counter(c.get("unanswerable_subtype")
                                             for c in cases if c["primary"] == "unanswerable")),
        "attributes": attributes,
        "no_system_fault": {"n": len(nofault), "share": round(len(nofault) / n, 4),
                            "ci_95": [round(lo, 4), round(hi, 4)], "cases": nofault},
        "category_by_coverage": by("coverage"),
        "category_by_stop_condition": by("stop_condition"),
        "category_by_question_type": {t: dict(Counter(c["primary"] for c in cases if c["type"] == t))
                                      for t in sorted({c["type"] for c in cases})},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render the Phase 7 failure-mode tables")
    p.add_argument("--no-chart", action="store_true",
                   help=f"skip writing the SVG chart to {CHART_PATH}")
    p.add_argument("--no-save", action="store_true",
                   help="print the table without writing to the results directory")
    a = p.parse_args()

    data = load()
    render_text(data)

    if not a.no_save:
        out = save(data, OUTPUT_PATH)
        print(f"\nFailure-mode counts written to {out}")

    if not a.no_chart:
        print(f"Chart written to {render_chart(data, CHART_PATH)}")
    print()