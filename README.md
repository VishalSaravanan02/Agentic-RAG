# Agentic Multi-Hop RAG

**MSc Dissertation — COMP702, University of Liverpool**
Student: Vishal Saravanan (201900556) · Supervisor: Dr Terry Payne

> ⚠️ Work in progress. This repository accompanies an ongoing MSc dissertation;
> the design and results are still being finalised.

---

## Overview

Standard Retrieval-Augmented Generation (RAG) retrieves documents **once** per
question. That works for simple lookups but fails on **multi-hop** questions,
where the answer requires chaining facts across several sources — the
information needed for a later search often isn't known until an earlier search
completes.

This project builds and evaluates an **agentic multi-hop RAG system** in which
every retrieval decision is made explicitly and adaptively, rather than being
fixed in advance. The system is compared against a standard single-hop RAG
baseline (and two further systems, below) on the
[HotpotQA](https://hotpotqa.github.io/) multi-hop question-answering benchmark.

## The four systems

The project compares four systems that share every component (corpus, chunking,
embedder, retriever, generation model, synthesis prompt) and differ **only** in
their retrieval strategy, so that any difference in results is attributable to
strategy alone:

| System | Retrieval strategy |
|---|---|
| **Baseline A** | Single-hop — retrieve once, then answer (standard RAG) |
| **Baseline B** | Fixed two-hop — always exactly two retrievals |
| **Ablation 1** | Adaptive retrieval, but **without** query decomposition |
| **Main System** | Full agentic pipeline (all five decision mechanisms) |

## The five decision mechanisms

The Main System makes each retrieval decision explicit:

1. **Hop necessity** — does this question need more than one retrieval?
2. **Query decomposition** — break the question into ordered sub-questions
3. **Sufficiency check** — is the retrieved evidence enough to answer?
4. **Adaptive stopping** — stop when sufficient, or at a hop ceiling
5. **Grounded synthesis** — answer using only the retrieved evidence

## Research questions

- **RQ1** — Does agentic multi-hop retrieval outperform single-hop RAG?
- **RQ2** — Does any advantage concentrate on questions the system itself
  classifies as multi-hop?
- **RQ3** — Which contributes more: adaptive retrieval decisions, or structured
  query decomposition?

## Repository structure

```
src/core/        Shared infrastructure (config, embedder, retriever, LLM client, logger)
src/agents/      The five decision mechanisms
src/systems/     The four systems + shared retrieval pipeline
src/evaluation/  Metrics
scripts/         Data prep, index building, evaluation runners, analysis
tests/           Unit / integration tests
results/         Per-system evaluation logs (JSON Lines)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11. API keys (OpenAI) are read from an untracked `.env` file.
The dataset and vector index are built locally via the scripts in `scripts/`
and are not committed to the repository.

## Tech stack

Python · ChromaDB (vector store) · sentence-transformers (`all-MiniLM-L6-v2`) ·
OpenAI API (`gpt-4o-mini`) · HotpotQA · pytest

## Status

All four systems are implemented and tested. Development-set evaluation is
complete; full-scale evaluation, significance testing, and LLM-as-judge scoring
are in progress.

## License

MIT — see [LICENSE](LICENSE).
