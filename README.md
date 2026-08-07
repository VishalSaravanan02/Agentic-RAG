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
  unit test enforces this so it cannot drift unnoticed.

## The five decision mechanisms

The Main System makes each retrieval decision explicit and independently
loggable:

1. **Hop necessity** — does this question need more than one retrieval?
2. **Query decomposition** — break the question into ordered sub-questions
3. **Sufficiency check** — is the retrieved evidence enough to answer?
4. **Adaptive stopping** — stop when sufficient, or at a four-hop ceiling
5. **Grounded synthesis** — answer from retrieved evidence only, or decline

## Results

All four systems evaluated on 1,000 held-out HotpotQA questions, run as a single
batch, 1000/1000 completed with zero failures.

| Metric | Baseline A | Baseline B | Ablation 1 | Main System |
|---|---|---|---|---|
| Exact Match | 0.3690 | **0.4520** | 0.4430 | 0.4230 |
| F1 | 0.4693 | **0.5820** | 0.5662 | 0.5429 |
| Supporting Facts Recall | 0.5775 | **0.7392** | 0.6999 | 0.6818 |
| Avg latency (ms) | **856** | 1,662 | 5,885 | 7,872 |
| Avg input tokens | **531** | 1,367 | 2,631 | 3,349 |

Significance testing throughout is paired bootstrap resampling, 10,000
resamples.

**RQ1 — Does agentic multi-hop retrieval beat single-hop RAG? Yes.**
The Main System answers 54 more questions correctly out of 1,000
(+0.0540 EM, p < 0.0002), gains 7.4 F1 points, and retrieves 10.4 additional
points of gold supporting evidence, at 9.2x the latency and 6.3x the input
tokens. An independent LLM-as-judge evaluation confirms its answers are more
relevant (+0.32 on a 1-5 scale, p = 0.0008).

**RQ2 — Does the advantage concentrate where the system predicts? Yes.**
The gain is larger on the 744 questions the hop classifier flags as multi-hop
(+0.0753 EM) than pooled (+0.0540). The classifier is validated independently of
the experimental structure: Baseline A, which never sees the classification,
itself scores higher on questions labelled single-hop (0.4023 EM) than on those
labelled multi-hop (0.3575). The gate discriminates on real differences in
question difficulty.

**RQ3 — Which agentic component contributes more? Neither, and one costs.**
Adaptive control flow showed no significant difference in answer quality against
a fixed two-hop pipeline (p = 0.2987). Query decomposition measurably *reduced*
performance (-0.0200 EM, p = 0.0220; -0.0233 F1, p = 0.0044). The simplest
multi-hop system tested is the most accurate.

## Root-cause analysis

The decomposition result is traced to a specific, measured mechanism rather than
left unexplained.

Baseline A, Baseline B and Ablation 1 all issue the **original question** as
their first retrieval query, and consequently retrieve byte-identical documents
at that step (verified 159/159). The Main System issues a decomposed sub-query
instead, matching on only 7/159 — and on 150 of 159 multi-hop questions it never
queries the original question at any point.

That first query matters. A HotpotQA bridge question implicitly references both
articles needed to answer it; a sub-query such as "Who is X?" can surface only
one. Measured as cumulative gold-evidence coverage by hop:

| System | hop 1 | 2 hops | 3 hops | 4 hops |
|---|---|---|---|---|
| Baseline B | 0.5570 | **0.7148** | — | — |
| Ablation 1 | 0.5570 | 0.6928 | 0.7001 | 0.7001 |
| Main System | **0.4097** | 0.5575 | 0.6466 | 0.6896 |

Decomposition forfeits 14.7 points of coverage at the first retrieval. The Main
System needs two hops to reach where the others start, and never closes the gap
within its budget.

Decomposition does confer a real benefit — it supplies the reactive retrieval
loop with query diversity, preventing the stalling that affects Ablation 1 in 56
of 81 budget-exhausted questions against the Main System's 8 of 82 — but on this
evidence the benefit does not offset the cost.

## Further findings

**The sufficiency check is well calibrated.** Prior work predicts that systems
judging their own retrieval sufficiency stop too early. This was tested rather
than assumed: the checker reports sufficiency at 0.81 gold-evidence coverage and
withholds it at 0.64. It discriminates correctly. The losses lie downstream, in
a reactive loop that keeps no record of queries already issued and therefore
reissues them until the budget is exhausted.

**The pipeline is not reproducible at temperature 0, and the noise floor is
measured.** On 256 questions the Main System and Baseline A execute provably
identical code, with retrieval verified identical in all 256 cases, yet 10 of
256 final answers differ. That is a 3.9% answer-level noise floor, obtained as a
free duplicate run, and it bounds what any single-run comparison can resolve.

**A demonstrable false positive.** On those same 256 questions, one metric
differs at p = 0.0482 between two systems that are provably identical. It is
retained in the write-up as a concrete argument for multiple-comparison
discipline: a p-value below 0.05 across roughly 40 tests means little without a
corresponding mechanism.

## Methodology notes

- **The evaluation split was never used during development.** All prompt work,
  debugging and defect repair was done against a separate 200-question
  development split. No significance testing was performed on the development
  data, so exploratory findings could not be reconciled against the confirmatory
  result after the fact.
- **The development ordering did not survive.** On 200 questions the Main System
  ranked first; on 1,000 it ranks third. The retrieval behaviour was consistent
  across both splits (gold-coverage figures within 0.02) — what changed was the
  statistical power to detect its consequences. The dissertation reports this
  divergence as a methodological result.
- **The bootstrap implementation was validated on synthetic inputs** whose
  correct answers are known in advance, rather than on project data. Four
  defects were found in review and fixed; each has a named regression test
  verified to fail when the fix is reverted.
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
tests/           44 unit and integration tests
results/         Per-system evaluation logs (JSON Lines)
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
python -m pytest tests/ -v                                        # 44 tests
```

Requires Python 3.11. API keys are read from an untracked `.env`. The dataset and
vector index (91,350 chunks) are built locally and are not committed.

## Tech stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · OpenAI API
(`gpt-4o-mini` for systems, `gpt-4o` for judging) · Groq API · scipy · pytest ·
HotpotQA

The agent loop, retrieval pipeline and evaluation harness are implemented
directly against these libraries rather than through an orchestration framework.

## Status

Complete: all four systems, full evaluation on 1,000 questions, significance
testing across answer quality, retrieval quality, efficiency and judge scores,
and root-cause analysis. Outstanding: a stratified 50-case qualitative failure
analysis, and the final dissertation write-up.

## License

MIT — see [LICENSE](LICENSE).