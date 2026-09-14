# Agentic Multi-Hop RAG

**MSc Dissertation — COMP702, University of Liverpool**
Vishal Saravanan · Supervisor: Dr Terry Payne

A controlled experimental study of whether agentic, decision-driven retrieval
improves multi-hop question answering, and which of its components actually earn
their cost.

---

## Overview

Standard Retrieval-Augmented Generation retrieves documents **once** per
question. That works for simple lookups but breaks down on **multi-hop**
questions, where the answer requires chaining facts across several sources: the
information needed for a later search often isn't known until an earlier search
completes.

This project builds an **agentic multi-hop RAG system** in which every retrieval
decision is made explicitly and adaptively, and evaluates it against three
comparison systems on 1,000 held-out questions from the
[HotpotQA](https://hotpotqa.github.io/) benchmark, with paired bootstrap
significance testing throughout.

The headline result is positive: the agentic system significantly outperforms
standard single-hop RAG. The more interesting result is negative, and is the
main contribution of the study — one of the agentic components measurably
*reduces* performance, and the project traces why.

## Experimental design

Four systems share every component (corpus, chunking, embedding model,
retriever, top-k, generation model, and synthesis prompt) and differ **only** in
retrieval strategy, so that any measured difference is attributable to strategy
alone.

| System | Retrieval strategy | Role |
|---|---|---|
| **Baseline A** | Single retrieval, then answer (standard RAG) | reference point |
| **Baseline B** | Exactly two retrievals, always | fixed multi-hop |
| **Ablation 1** | Adaptive retrieval, **no** query decomposition | isolates adaptive control flow |
| **Main System** | Full agentic pipeline, all five mechanisms | isolates decomposition |

Two design choices make the comparisons structurally sound rather than merely
intended:

- The Main System and Ablation 1 execute the **same retrieval module** behind a
  single `use_decomposition` flag, so their comparison isolates decomposition by
  construction rather than by convention.
- The answer-synthesis prompt is byte-identical across all four systems, and a
  test enforces this so it cannot drift unnoticed.

## The five decision mechanisms

The Main System makes each retrieval decision explicit and independently
loggable:

1. **Hop necessity** — does this question need more than one retrieval?
2. **Query decomposition** — break the question into ordered sub-questions
3. **Sufficiency check** — is the retrieved evidence enough to answer?
4. **Adaptive stopping** — stop when sufficient, or at a four-hop ceiling
5. **Grounded synthesis** — answer from retrieved evidence only, or decline

## Results

All four systems were evaluated on the same 1,000 held-out HotpotQA questions,
on the same day and against the same model version, 1000/1000 completed with
zero pipeline failures. Baseline A's run was interrupted after 869 questions by
a credit exhaustion and resumed from its checkpoint 8h22m later; the remaining
131 questions took under three minutes.

| Metric | Baseline A | Baseline B | Ablation 1 | Main System |
|---|---|---|---|---|
| Exact Match | 0.3690 | **0.4520** | 0.4430 | 0.4230 |
| F1 | 0.4693 | **0.5820** | 0.5662 | 0.5429 |
| Supporting Facts Recall | 0.5745 | **0.7375** | 0.6975 | 0.6810 |
| Recall@k | 0.8550 | 0.9020 | 0.8870 | **0.9060** |
| Duplicate Retrieval Rate | **0.0000** | 0.1495 | 0.2253 | 0.1765 |
| Avg latency (ms) | **856** | 1,662 | 5,885 | 7,872 |
| Avg input tokens | **531** | 1,367 | 2,630 | 3,350 |

Significance testing throughout is paired bootstrap resampling, 10,000
resamples, seed 42.

**RQ1 — Does agentic multi-hop retrieval beat single-hop RAG? Yes.**
The Main System answers 54 more questions correctly out of 1,000
(+0.0540 EM, p < 0.0002), gains 7.4 F1 points, and retrieves 10.7 additional
points of gold supporting evidence, at 9.2x the latency and 6.3x the input
tokens. An LLM-as-judge evaluation scores its answers as more relevant
(+0.32 on a 1–5 scale, p = 0.0008).

**RQ2 — Does the advantage concentrate where the system predicts? Yes, but the
concentration is largely architectural.**
The gain is significantly larger on the 744 questions the hop classifier flags
as multi-hop than on the 256 it does not (+0.0831 EM interaction, p < 0.0002).
That the concentration is real does not establish that the classifier is
identifying genuinely harder questions: on the single-hop-classified subset the
Main System and Baseline A execute identical code, so a zero gap there is
guaranteed by construction rather than measured. Baseline A's own advantage on
that subset is not significant (+0.0448 EM, p = 0.21), and Baseline B — which
ignores the classification entirely — gains nearly as much from a second hop on
those questions as on the multi-hop ones (+0.0703 against +0.0874, interaction
p = 0.47). The classifier is therefore withholding retrieval from questions that
would have benefited from it.

**RQ3 — Which agentic component contributes more? Neither, and one costs.**
Adaptive control flow showed no significant difference in answer quality against
a fixed two-hop pipeline (−0.0090 EM, p = 0.2987). Query decomposition
measurably *reduced* performance (−0.0200 EM, p = 0.0220; −0.0233 F1,
p = 0.0044). The simplest multi-hop system tested is the most accurate.

## Root-cause analysis

The decomposition result is traced to a specific, measured mechanism rather than
left unexplained.

Baseline A, Baseline B and Ablation 1 all issue the **original question** as
their first retrieval query, and consequently retrieve identical documents at
that step. The Main System issues a decomposed sub-query instead, and on the
great majority of multi-hop questions it never queries the original question at
any point. (These first-query identity checks were run on the 200-question
development split: 159/159 identical for the comparison systems, 7/159 for the
Main System, and 150/159 never issuing the original question at all.)

That first query matters. A HotpotQA bridge question implicitly references both
articles needed to answer it; a sub-query such as "Who is X?" can surface only
one. Measured as cumulative gold-evidence coverage by hop, over the 744
multi-hop-classified questions of the evaluation split:

| System | hop 1 | 2 hops | 3 hops | 4 hops |
|---|---|---|---|---|
| Baseline A | 0.5719 | — | — | — |
| Baseline B | 0.5719 | **0.7460** | — | — |
| Ablation 1 | 0.5719 | 0.7124 | 0.7272 | 0.7339 |
| Main System | **0.4308** | 0.5733 | 0.6593 | 0.7151 |

Decomposition forfeits 14.1 points of coverage at the first retrieval. The Main
System needs two hops to reach where the others start, and never closes the gap
within its four-hop budget.

Decomposition does confer a real benefit — it supplies the reactive retrieval
loop with query diversity, preventing the stalling that affects Ablation 1 in
245 of its 403 budget-exhausted questions against the Main System's 36 of 415 —
but on this evidence the benefit does not offset the cost. (A repeat is a query
exactly matching one issued earlier for the same question; measured over all
reactive queries, the repeat rates are 24% for Ablation 1 and 8% for the Main
System.)

## Further findings

**The sufficiency check discriminates, but stops early.** Prior work predicts
that systems judging their own retrieval sufficiency stop too early. This was
tested rather than assumed, and the prediction holds in a specific form. The
checker does discriminate: it reports sufficiency at 0.81 gold-evidence coverage
and withholds it at 0.64. But 37.1% of the 329 sufficiency stops occurred with a
gold article still missing, and 68.3% of those premature stops happened at the
first hop — the threshold is set too low at hop 1 rather than being uncalibrated
in general. Against a counterfactual fixed two-hop policy the stopping rule is
roughly net-neutral: about 10 questions lost by stopping early, about 10 gained
by stopping correctly.

**The pipeline is not reproducible at temperature 0, and the noise floor is
measured.** On 256 questions the Main System and Baseline A execute provably
identical code, with retrieval verified identical in all 256 cases, yet 10 of
256 final answers differ. That is a 3.9% answer-level noise floor, obtained as a
free duplicate run, and it bounds what any single-run comparison can resolve.

**A demonstrable false positive.** On those same 256 questions, one metric
differs at p = 0.0482 between two systems that are provably identical. It is
retained in the write-up as a concrete argument for multiple-comparison
discipline: a p-value below 0.05 across 36 quantitative tests means little
without a corresponding mechanism.

**Fifty failures were coded by hand, and one prediction was retested.** A
stratified sample of 50 Main System failures was categorised against a
seven-category taxonomy. Reading them suggested that the decomposition penalty
should fall on bridge questions and not on comparison questions. Because the 50
coded cases are themselves inside the evaluation 1,000, the prediction was
retested on the other 950: it holds, with a bridge penalty of −0.0252 EM
(p = 0.0127) against no penalty on comparison questions, and a
bridge-minus-comparison interaction of p = 0.0406 on Exact Match — though
p = 0.0944 on F1, which is reported as not significant.

## Methodology notes

- **The evaluation split was never used during development.** All prompt work,
  debugging and defect repair was done against a separate 200-question
  development split. No significance testing was performed on the development
  data, so exploratory findings could not be reconciled against the confirmatory
  result after the fact.
- **The development ordering did not survive.** On 200 questions the Main System
  ranked first; on 1,000 it ranks third. The retrieval behaviour was consistent
  across both splits — what changed was the statistical power to detect its
  consequences. The dissertation reports this divergence as a methodological
  result.
- **The bootstrap implementation was validated on synthetic inputs** whose
  correct answers are known in advance, rather than on project data. Defects
  found in review were fixed, and each carries a named regression test verified
  to fail when the fix is reverted.
- **Three defects in the evaluation code are documented rather than repaired,**
  so that the committed code is the code that produced the results. A
  sensitivity analysis excluding every affected question leaves all conclusions
  unchanged in direction.
- **All four systems were frozen** before the final evaluation. An obvious
  improvement follows from the root-cause analysis above; it was deliberately
  not implemented, and is recorded as future work.

## Repository structure

```
src/core/        Shared infrastructure (config, embedder, retriever, LLM client, logger)
src/agents/      The five decision mechanisms
src/systems/     The four systems and the shared retrieval pipeline
src/evaluation/  Metrics, paired bootstrap, LLM-as-judge
scripts/         Data preparation, index building, evaluation runners, analysis
tests/           75 contract, integration, invariant and statistical tests
results/         Per-system evaluation logs (JSON Lines) and analysis outputs
```

Every run writes a fixed schema per question — decision outputs, hops taken, the
query issued at each hop, documents retrieved per hop, stop condition, token
counts and latency — so any result in the dissertation can be traced back to the
decision trail that produced it. Evaluation runs are checkpointed and resume
from the last completed question.

## Setup

```bash
pip install -r requirements.txt

python scripts/download_data.py                                   # fetch and sample HotpotQA
python scripts/build_index.py                                     # build the ChromaDB index
python scripts/run_evaluation.py --system main_system --split dev # run a system
python -m pytest tests/ -v                                        # 75 tests
```

Requires Python 3.11. API keys are read from an untracked `.env`. The corpus
(66,581 articles, chunked into 91,350 chunks) and the vector index are built
locally and are not committed.

## Tech stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · OpenAI API
(`gpt-4o-mini` for systems, `gpt-4o` for judging) · Groq API · scipy · pytest ·
HotpotQA

The agent loop, retrieval pipeline and evaluation harness are implemented
directly against these libraries rather than through an orchestration framework.

## Status

Complete: all four systems, full evaluation on 1,000 questions, significance
testing across answer quality, retrieval quality, efficiency and judge scores,
root-cause analysis, and a stratified 50-case qualitative failure analysis with
a held-out retest of its central prediction.

## License

MIT — see [LICENSE](LICENSE).