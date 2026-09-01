# =============================================================================
# demo.py — Live walkthrough of one question, decision by decision.
#
# Built for the project video: it prints each agent decision as it happens, so
# the retrieval loop can be watched rather than described. Then it runs the same
# question through Baseline A for contrast.
#
# It does NOT reimplement the pipeline. It wraps the four decision functions in
# _shared_retrieval with printing versions and calls the real systems, so what
# you see on screen is the actual evaluated code path — nothing is staged.
#
# Usage
#   python scripts/demo.py                      # a bridge question from eval
#   python scripts/demo.py --qid 5abe36745542991f66106101   # a specific question
#   python scripts/demo.py --index 7            # a different eval question
#   python scripts/demo.py --question "..."     # your own question
#   python scripts/demo.py --solo               # Main System only, no comparison
#
# The Main System is narrated decision by decision. The other three are then run
# on the same question and reported compactly, so the four-system design is
# visible without four full narrations.
#
# NOTE: one question is an illustration, not a result. Say "on this question",
# not "as you can see, X is better" — a single case proves nothing either way.
# =============================================================================

import argparse
import json
import os
import sys
import time

# Keep the HF progress bars and token warning off the screen during recording.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import EVAL_DATA_PATH, EVAL_MODEL
from src.evaluation.metrics import exact_match, f1_score
import src.systems._shared_retrieval as sr
from src.systems.main_system import run_main_system
from src.systems.baseline_a import run_baseline_a
from src.systems.baseline_b import run_baseline_b
from src.systems.ablation_1 import run_ablation_1

# ── Presentation formatting ─────────────────────────────────────────────────
W = 78
DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"
AMBER, TEAL, GREY = "\033[33m", "\033[36m", "\033[90m"

PAUSE = 0.0   # set with --pause to slow the output down for recording


def rule(char="─"):
    print(GREY + char * W + OFF)


def banner(text):
    print()
    rule("━")
    print(f"{BOLD}{text}{OFF}")
    rule("━")


def decision(tag, name, verdict, detail=None):
    """One agent decision, formatted so it reads on screen."""
    print()
    print(f"{AMBER}{BOLD}  [{tag}] {name}{OFF}   →   {BOLD}{verdict}{OFF}")
    if detail:
        for line in detail:
            print(f"       {DIM}{line}{OFF}")
    time.sleep(PAUSE)


def step(text):
    print(f"\n{TEAL}  ▸ {text}{OFF}")
    time.sleep(PAUSE)


# ── Wrap the real decision functions so each one narrates itself ────────────
_classify, _decompose, _check, _retrieve = sr.classify, sr.decompose, sr.check, sr.retrieve
_hop = {"n": 0}


def classify_v(question, **kw):
    r = _classify(question, **kw)
    yes = r["classification"] == "YES"
    decision("D1", "Hop necessity",
             "YES — needs multi-hop" if yes else "NO — one search is enough",
             ["Does answering this require more than one retrieval?"])
    return r


def decompose_v(question, **kw):
    r = _decompose(question, **kw)
    decision("D2", "Decomposition", f"{len(r['sub_questions'])} sub-questions",
             [f"{i+1}. {q}" for i, q in enumerate(r["sub_questions"])])
    return r


def retrieve_v(query, k=5, **kw):
    _hop["n"] += 1
    step(f"HOP {_hop['n']} — searching for: \"{query}\"")
    docs = _retrieve(query, k=k, **kw)
    for d in docs:
        print(f"       {d['cosine_similarity']:.3f}  {d['metadata']['article_title']}")
    time.sleep(PAUSE)
    return docs


def check_v(question, context, **kw):
    r = _check(question, context, **kw)
    if r["sufficient"]:
        decision("D3", "Sufficiency check", "YES — enough to answer",
                 ["Stopping here, before the plan is exhausted."])
    else:
        # The raw response carries a verdict, a "Present:" summary and a
        # "MISSING:" query. Show only the gap — that is what drives the next
        # hop, and the rest is noise on screen.
        raw = (r.get("missing") or "").replace("\n", " ")
        for key in ("MISSING:", "Missing:"):
            if key in raw:
                raw = raw.rsplit(key, 1)[1]
                break
        gap = " ".join(raw.split())
        if len(gap) > 110:                      # trim on a word boundary
            gap = gap[:110].rsplit(" ", 1)[0] + "…"
        decision("D3", "Sufficiency check", "NO — keep going",
                 [f"Still needs: {gap}"])
    return r


def install_narration():
    sr.classify, sr.decompose, sr.check, sr.retrieve = (
        classify_v, decompose_v, check_v, retrieve_v)


def remove_narration():
    sr.classify, sr.decompose, sr.check, sr.retrieve = (
        _classify, _decompose, _check, _retrieve)


# ── Question selection ──────────────────────────────────────────────────────
def pick_question(index, custom, qid):
    if custom:
        return {"id": "demo", "question": custom, "answer": "(unknown)",
                "supporting_facts": {"title": []}}
    if not os.path.exists(EVAL_DATA_PATH):
        raise SystemExit(f"Question file not found: {EVAL_DATA_PATH}\n"
                         f"Pass --question \"...\" instead.")
    qs = json.load(open(EVAL_DATA_PATH))
    if qid:
        for q in qs:
            if q["id"] == qid:
                return q
        raise SystemExit(f"No question with id {qid} in {EVAL_DATA_PATH}")
    bridges = [q for q in qs if q.get("type") == "bridge"]
    pool = bridges or qs
    return pool[index % len(pool)]


def main(index, custom, qid, model, show_baseline, pause):
    global PAUSE
    PAUSE = pause
    q = pick_question(index, custom, qid)

    # Load the embedding model now, so its startup messages don't appear in the
    # middle of the narrated run.
    print(f"{DIM}Loading embedding model...{OFF}")
    from src.core.embedder import embed
    embed("warm up")
    print("\033[F\033[K", end="")   # erase that line

    banner("QUESTION")
    print(f"  {q['question']}")
    if q.get("answer") != "(unknown)":
        print(f"\n  {DIM}Gold answer: {q['answer']}{OFF}")
        gold = q.get("supporting_facts", {}).get("title", [])
        if gold:
            print(f"  {DIM}Required evidence: {', '.join(gold)}{OFF}")

    # ── Main System, narrated ───────────────────────────────────────────────
    banner("MAIN SYSTEM — every decision made inside the retrieval loop")
    _hop["n"] = 0
    install_narration()
    t0 = time.time()
    try:
        result = run_main_system(q["question"], q["id"], q.get("answer", ""), model=model)
    finally:
        remove_narration()
    elapsed = time.time() - t0

    print()
    rule()
    print(f"{BOLD}  ANSWER:{OFF} {result['final_answer']}")
    print(f"  {DIM}{result['num_hops']} hop(s) · stopped on '{result['stop_condition_triggered']}' "
          f"· {elapsed:.1f}s · {result['input_tokens']:,} in / {result['output_tokens']:,} out{OFF}")
    rule()

    if not show_baseline:
        return

    # ── The other three systems, compactly ──────────────────────────────────
    banner("THE SAME QUESTION, THROUGH THE OTHER THREE SYSTEMS")

    others = [
        ("Baseline A",  "one retrieval, no decisions",        run_baseline_a),
        ("Baseline B",  "exactly two retrievals, fixed",      run_baseline_b),
        ("Ablation 1",  "full agent, no decomposition",       run_ablation_1),
    ]
    rows = [("Main System", "all five mechanisms", result, elapsed)]

    for name, desc, fn in others:
        t = time.time()
        r = fn(q["question"], q["id"], q.get("answer", ""), model=model)
        el = time.time() - t
        rows.append((name, desc, r, el))
        print(f"\n{BOLD}  {name}{OFF}  {DIM}— {desc}{OFF}")
        print(f"       {r['num_hops']} hop(s) · {el:.1f}s")
        print(f"       {BOLD}{r['final_answer']}{OFF}")
        time.sleep(PAUSE)

    # ── One-line comparison ─────────────────────────────────────────────────
    gold = q.get("answer", "")
    if gold and gold != "(unknown)":
        print()
        rule("━")
        print(f"{BOLD}  ON THIS QUESTION{OFF}   {DIM}(one case — an illustration, not a result){OFF}")
        rule()
        print(f"  {'system':<14}{'hops':>5}{'time':>8}{'EM':>5}   answer")
        for name, _, r, el in sorted(rows, key=lambda x: x[0]):
            em = exact_match(r["final_answer"], gold)
            mark = "✓" if em else "·"
            ans = r["final_answer"][:34]
            print(f"  {name:<14}{r['num_hops']:>5}{el:>7.1f}s{mark:>5}   {ans}")
        rule("━")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Narrated single-question demo")
    p.add_argument("--index", type=int, default=0, help="which bridge question from the eval set")
    p.add_argument("--qid", type=str, default=None, help="run a specific question by HotpotQA id")
    p.add_argument("--question", type=str, default=None, help="use your own question instead")
    p.add_argument("--model", type=str, default=EVAL_MODEL)
    p.add_argument("--solo", action="store_true",
                   help="Main System only — skip the three-system comparison")
    p.add_argument("--pause", type=float, default=0.0,
                   help="seconds to pause after each step (try 0.6 for recording)")
    a = p.parse_args()
    main(a.index, a.question, a.qid, a.model, not a.solo, a.pause)