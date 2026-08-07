# =============================================================================
# llm_judge.py — LLM-as-judge scoring (proposal Section 7.3)
#
# Scores answers on faithfulness, relevance and coherence (1-5 each) using a
# frontier-tier judge, on a fixed 200-question subsample shared by all systems.
#
# WHY CONTEXT IS RECONSTRUCTED RATHER THAN READ FROM THE LOGS:
# the result schema records retrieved chunk IDs, not chunk text, so the text the
# system actually saw must be re-fetched from ChromaDB. The logged ID format is
# "{article_title}_{chunk_index}" while ChromaDB's own IDs are sequential
# ("chunk_0", "chunk_1", ...), so lookup goes through metadata, not IDs.
#
# Reconstruction reproduces _shared_retrieval.py exactly: dedup by
# (article_title, chunk_index) preserving hop order, format each chunk as
# "[title] text", join with blank lines. The context budget is NOT replayed,
# because it drops the lowest-similarity chunks and cosine similarity is not
# logged. This is safe as long as the budget never binds — verified across all
# four systems on dev (largest context 2,527 tokens against a 3,000 limit) —
# and run_judge() warns per question if it ever would.
# =============================================================================

import json
import os
import random
import re

from src.core.config import (
    JUDGE_MODEL,
    MAX_CONTEXT_TOKENS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.core.llm_client import call_llm
from src.core.logger import load_results
from src.core.retriever import get_collection

JUDGE_SAMPLE_SIZE = 200
MAX_RETRIES = 3
DIMENSIONS = ("faithfulness", "relevance", "coherence")

# -----------------------------------------------------------------------------
# Fixed prompt template — identical for every system and every question.
# Reproduce verbatim in the dissertation appendix so scoring can be audited.
# -----------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are evaluating an answer produced by a retrieval-augmented question answering system.

QUESTION:
{question}

RETRIEVED CONTEXT (the only evidence the system was given):
{context}

SYSTEM ANSWER:
{answer}

Score the answer on three dimensions, each from 1 to 5.

FAITHFULNESS — is every claim in the answer supported by the retrieved context?
5 = every claim is directly supported by the context
3 = mostly supported, with some unsupported detail
1 = contradicts the context, or asserts facts absent from it
An answer that states the evidence is insufficient, and does not assert anything beyond it, is FAITHFUL and should score 5.

RELEVANCE — does the answer address the question that was asked?
5 = directly answers the question asked
3 = partially addresses it, or answers a related but different question
1 = does not address the question
An answer that declines for lack of evidence has NOT addressed the question and should score low here, however faithful it is.

COHERENCE — is the answer well-formed, clear and internally consistent?
5 = clear and internally consistent
3 = understandable but awkward or partly unclear
1 = incoherent or self-contradictory

Judge only against the retrieved context, not your own knowledge of the subject. Do not reward length: a short exact answer is not worse than a long one, and additional wording that adds no supported content should not raise any score.

Respond with JSON only, no other text, in exactly this form:
{{"faithfulness": <1-5>, "relevance": <1-5>, "coherence": <1-5>}}"""


# --- Context reconstruction --------------------------------------------------

def parse_doc_id(doc_id: str) -> tuple[str, int]:
    """
    Split a logged chunk ID into (article_title, chunk_index).

    Splits on the LAST underscore, so titles containing underscores survive:
    "Foo_2_3" -> ("Foo_2", 3).
    """
    title, _, index = doc_id.rpartition("_")
    if not title or not index.isdigit():
        raise ValueError(f"malformed chunk id: {doc_id!r}")
    return title, int(index)


def fetch_chunk_texts(doc_ids: list[str]) -> dict:
    """
    Fetch chunk text from ChromaDB for a set of logged chunk IDs.

    Queries by article_title metadata (one batched call for all titles), then
    keys the result by (title, chunk_index). Missing chunks are simply absent
    from the returned dict; callers decide how to handle that.
    """
    if not doc_ids:
        return {}

    titles = sorted({parse_doc_id(d)[0] for d in doc_ids})
    collection = get_collection()

    where = ({"article_title": titles[0]} if len(titles) == 1
             else {"$or": [{"article_title": t} for t in titles]})
    found = collection.get(where=where, include=["documents", "metadatas"])

    lookup = {}
    for text, meta in zip(found["documents"], found["metadatas"]):
        lookup[(meta["article_title"], meta["chunk_index"])] = text
    return lookup


def reconstruct_context(result: dict) -> tuple[str, list[str]]:
    """
    Rebuild the context string a system was given, from its logged chunk IDs.

    Mirrors _shared_retrieval.py: dedup by (article_title, chunk_index) in hop
    order, format as "[title] text", join with blank lines.

    Returns (context, missing_ids). A non-empty missing_ids means some chunk
    could not be found in ChromaDB and the context is incomplete — the caller
    should skip that question rather than judge a partial context.
    """
    flat = [d for hop in result["docs_retrieved_per_hop"] for d in hop]
    lookup = fetch_chunk_texts(flat)

    seen, parts, missing = set(), [], []
    for doc_id in flat:
        title, index = parse_doc_id(doc_id)
        if (title, index) in seen:
            continue
        seen.add((title, index))
        text = lookup.get((title, index))
        if text is None:
            missing.append(doc_id)
            continue
        parts.append(f"[{title}] {text}")

    return "\n\n".join(parts), missing


def estimate_context_tokens(context: str) -> float:
    """Word count times 1.33, matching _shared_retrieval.py's estimator."""
    return len(context.split()) * 1.33


# --- Judging -----------------------------------------------------------------

def _parse_scores(raw: str) -> dict:
    """
    Extract the three scores from a judge response.

    Tolerates fenced code blocks and stray prose around the JSON, but requires
    all three dimensions present as integers in 1-5. Anything else raises, so
    the caller retries rather than silently recording a malformed score.
    """
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge response")

    data = json.loads(match.group(0))
    scores = {}
    for dim in DIMENSIONS:
        if dim not in data:
            raise ValueError(f"missing dimension: {dim}")
        value = data[dim]
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError(f"{dim} is not an integer in 1-5: {value!r}")
        scores[dim] = value
    return scores


def judge_answer(question: str, context: str, answer: str,
                 model: str = JUDGE_MODEL) -> dict:
    """
    Score one answer. Retries up to MAX_RETRIES on malformed output, matching
    the validate-and-retry convention used by the five decision mechanisms.

    Raises after the final attempt rather than returning a default: a fabricated
    score is worse than a recorded failure.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context, answer=answer
    )

    last_error = None
    for _ in range(MAX_RETRIES):
        response = call_llm(prompt, model=model, max_tokens=100, temperature=0.0)
        try:
            scores = _parse_scores(response["text"])
            scores["input_tokens"] = response["input_tokens"]
            scores["output_tokens"] = response["output_tokens"]
            return scores
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e

    raise ValueError(f"judge returned unparseable output {MAX_RETRIES} times: {last_error}")


# --- Sampling ----------------------------------------------------------------

def get_judge_sample(split: str, systems: list[str],
                     n: int = JUDGE_SAMPLE_SIZE,
                     seed: int = RANDOM_SEED) -> list[str]:
    """
    Draw the fixed question sample, identical for every system.

    Sampled from questions ALL systems answered, so judge scores stay paired and
    can be compared with the same paired bootstrap used for the other metrics.
    Sorting before sampling makes the draw independent of file order.
    """
    shared = None
    for system in systems:
        ids = {r["question_id"] for r in load_results(system, split)}
        if not ids:
            raise ValueError(f"no results found for {system} on split '{split}'")
        shared = ids if shared is None else shared & ids

    if len(shared) < n:
        raise ValueError(
            f"only {len(shared)} questions common to all systems; need {n}"
        )

    return random.Random(seed).sample(sorted(shared), n)


# --- Run ---------------------------------------------------------------------

def _judge_filepath(system_name: str, split: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"judge_{system_name}_{split}.jsonl")


def load_judge_results(system_name: str, split: str) -> list[dict]:
    path = _judge_filepath(system_name, split)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_judge(system_name: str, split: str, question_ids: list[str],
              model: str = JUDGE_MODEL, verbose: bool = True) -> int:
    """
    Judge one system over the sampled questions, appending to
    results/judge_{system}_{split}.jsonl.

    Resumes from whatever is already scored, so an interrupted run can be
    restarted without repeating (and re-paying for) completed questions.
    Returns the number of questions scored in this invocation.
    """
    done = {r["question_id"] for r in load_judge_results(system_name, split)}
    results = {r["question_id"]: r for r in load_results(system_name, split)}
    path = _judge_filepath(system_name, split)

    todo = [q for q in question_ids if q not in done]
    if verbose:
        print(f"{system_name}/{split}: {len(todo)} to judge "
              f"({len(done)} already done)")

    scored = 0
    for i, qid in enumerate(todo, 1):
        if qid not in results:
            print(f"  SKIP {qid}: not in {system_name} results")
            continue

        r = results[qid]
        context, missing = reconstruct_context(r)
        if missing:
            print(f"  SKIP {qid}: {len(missing)} chunk(s) not found in ChromaDB "
                  f"(e.g. {missing[0]!r}) — context would be incomplete")
            continue
        if estimate_context_tokens(context) > MAX_CONTEXT_TOKENS:
            print(f"  WARN {qid}: reconstructed context ~"
                  f"{estimate_context_tokens(context):.0f} tokens exceeds the "
                  f"{MAX_CONTEXT_TOKENS} budget; the system may have truncated "
                  f"it, so this context may not match what it actually saw")

        scores = judge_answer(r["question"], context, r["final_answer"], model=model)

        row = {
            "question_id": qid,
            "system_name": system_name,
            "split": split,
            "judge_model": model,
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "coherence": scores["coherence"],
            "input_tokens": scores["input_tokens"],
            "output_tokens": scores["output_tokens"],
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

        scored += 1
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(todo)}")

    return scored


def judge_means(system_name: str, split: str) -> dict:
    """Mean score per dimension, for reporting."""
    rows = load_judge_results(system_name, split)
    if not rows:
        return {}
    out = {"n": len(rows)}
    for dim in DIMENSIONS:
        out[dim] = sum(r[dim] for r in rows) / len(rows)
    return out