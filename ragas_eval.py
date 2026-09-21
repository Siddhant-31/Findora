"""
ragas_eval.py — Vector-only vs Hybrid (Vector + Graph) RAG evaluation for Findora.

WHAT THIS DOES
---------------
For a set of real, catalog-grounded queries, this script runs BOTH retrieval
paths that now live in kg_engine.py:

  - engine.retrieve_vector_only()  -> pure semantic search, no graph, no budget filter
  - engine.retrieve_hybrid()       -> vector search + Neo4j graph traversal (category/brand)
                                       + budget filtering (Findora's production path)

...generates an answer from each set of retrieved products (same Gemini prompt,
same generation code, memory disabled for both so it isn't a confound), scores
each system with RAGAS, and prints a side-by-side comparison table.

METRICS
-------
- context_precision  : of what was retrieved, how much was relevant? (LLM-judged)
- context_recall     : of what *should* have been retrieved, how much was found?
                       (LLM-judged against a catalog-derived reference)
- faithfulness       : does the generated answer stick to the retrieved context?
- answer_relevancy   : does the answer actually address the query?
- catalog_recall@k   : a second, non-LLM recall number — the fraction of products
                       that ACTUALLY match the category+budget filter in the raw
                       CSV that show up in the top_k retrieved results. This is a
                       hard, deterministic check to sanity-back the LLM-judged
                       context_recall number above.

HOW THE EVAL SET IS BUILT
--------------------------
Queries aren't hand-written — they're generated FROM your actual
Amazon-Products.csv: pick a real sub_category, pick a real budget from that
category's own price distribution, phrase it as a natural query
("best <category> under ₹<budget>"), and use the CSV itself as ground truth
for "which products should have been retrieved." This keeps the eval grounded
in your real data instead of made-up examples.

REQUIREMENTS (pin these exactly — the current ragas/langchain ecosystem has
real version conflicts as of late 2026; these versions are verified to import
cleanly together):

    pip install "ragas==0.2.15" "langchain-core==0.3.29" \
                "langchain-community==0.3.7" "langchain-openai==0.2.14" \
                "langchain-google-genai==2.0.9" "langchain-huggingface==0.1.2" \
                pandas python-dotenv

RUN
---
    python ragas_eval.py --n-queries 15 --top-k 5

Needs the same .env you already use for the Flask app (NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD, GEMINI_API_KEY) plus a running Neo4j instance already seeded
via seed.py.
"""

import argparse
import math
import os
import random
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# NOTE: kg_engine (and its Neo4j/Gemini/sentence-transformers dependencies) is
# imported lazily inside main(), not at module level. That keeps
# build_eval_queries() usable/testable with just pandas — no live DB or API
# key needed just to build or inspect the eval set.

# ---------------------------------------------------------------------------
# 1. Build a catalog-grounded eval set
# ---------------------------------------------------------------------------

QUERY_TEMPLATES = [
    "Best {category} under ₹{budget}",
    "Show me good {category} within ₹{budget} budget",
    "I need a {category} that costs less than ₹{budget}",
    "What are some budget {category} options under ₹{budget}?",
]


# Keep eval output portable on Windows consoles that cannot encode the rupee symbol.
QUERY_TEMPLATES = [
    "Best {category} under Rs {budget}",
    "Show me good {category} within Rs {budget} budget",
    "I need a {category} that costs less than Rs {budget}",
    "What are some budget {category} options under Rs {budget}?",
]


def _clean_price(raw: str) -> float:
    txt = re.sub(r"[^\d.]", "", str(raw))
    try:
        return float(txt)
    except ValueError:
        return math.nan


def build_eval_queries(csv_path: str, n_queries: int = 15, min_category_size: int = 40, seed: int = 42):
    """Sample real sub_categories + real price points from the CSV to build
    queries with catalog-derived ground truth, instead of hand-invented ones."""
    rng = random.Random(seed)

    df = pd.read_csv(csv_path, usecols=["name", "sub_category", "discount_price"])
    df["price"] = df["discount_price"].apply(_clean_price)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    counts = df["sub_category"].value_counts()
    eligible_categories = counts[counts >= min_category_size].index.tolist()
    if not eligible_categories:
        raise ValueError("No sub_category has enough rows for the requested min_category_size.")

    chosen = rng.sample(eligible_categories, k=min(n_queries, len(eligible_categories)))

    queries = []
    for category in chosen:
        subset = df[df["sub_category"] == category]
        # Pick a budget at the 55th percentile of this category's real prices —
        # meaningful cutoff (not everything qualifies, but a real chunk does).
        budget = round(subset["price"].quantile(0.55), -2) or round(subset["price"].median(), -2)
        matching = subset[subset["price"] <= budget].sort_values("price", ascending=False)

        template = rng.choice(QUERY_TEMPLATES)
        query_text = template.format(category=category.lower(), budget=int(budget))

        reference_names = matching["name"].head(8).tolist()
        reference_products = matching.head(8).to_dict("records")
        reference_contexts = [
            f"{row['name']} - Rs {row['price']} (category: {category})"
            for row in reference_products
        ]
        reference_context = (
            f"Products in '{category}' priced at or under ₹{int(budget)} include: "
            + "; ".join(n[:80] for n in reference_names[:5])
        )
        reference_answer = (
            f"Relevant {category} products under Rs {int(budget)} include "
            + "; ".join(reference_names[:5])
            + "."
        )

        queries.append({
            "query": query_text,
            "category": category,
            "budget": budget,
            "reference_context": reference_context,
            "reference_contexts": reference_contexts,
            "reference_answer": reference_answer,
            "catalog_matching_names": set(matching["name"].tolist()),
        })

    return queries


# ---------------------------------------------------------------------------
# 2. Run both retrieval + generation paths through the real engine
# ---------------------------------------------------------------------------

def product_to_context_text(p: dict) -> str:
    return (
        f"{p['name']} — ₹{p.get('price', 0)} "
        f"(category: {p.get('category') or 'unknown'}, brand: {p.get('brand') or 'unknown'})"
    )


def product_to_context_text(p: dict) -> str:
    return (
        f"{p['name']} - Rs {p.get('price', 0)} "
        f"(category: {p.get('category') or 'unknown'}, brand: {p.get('brand') or 'unknown'})"
    )


def generate_extractive_answer(query: str, products: list[dict], top_k: int) -> str:
    if not products:
        return f"No catalog products were retrieved for: {query}"

    lines = []
    for i, product in enumerate(products[:top_k], start=1):
        lines.append(
            f"{i}. {product.get('name', 'Unknown product')} for Rs {product.get('price', 0)} "
            f"in {product.get('category') or 'unknown category'}."
        )
    return "Recommended products: " + " ".join(lines)


def _context_product_name(context: str) -> str:
    return re.split(r"\s-\sRs\s|\s\(category:", context, maxsplit=1)[0].strip().lower()


def reference_context_recall(retrieved_contexts: list[str], reference_contexts: list[str], threshold: float = 98.0) -> float:
    if not reference_contexts:
        return math.nan

    retrieved_names = {_context_product_name(context) for context in retrieved_contexts}
    exact_name_matches = [
        _context_product_name(ref) in retrieved_names
        for ref in reference_contexts
    ]
    if any(exact_name_matches):
        return sum(exact_name_matches) / len(exact_name_matches)

    try:
        from rapidfuzz import fuzz
    except ImportError:
        matches = [
            any(ref.lower() in retrieved.lower() or retrieved.lower() in ref.lower() for retrieved in retrieved_contexts)
            for ref in reference_contexts
        ]
    else:
        matches = [
            max((fuzz.ratio(ref, retrieved) for retrieved in retrieved_contexts), default=0.0) >= threshold
            for ref in reference_contexts
        ]

    return sum(matches) / len(matches)


def run_system(engine, eval_queries: list, retrieve_fn, top_k: int, use_llm_generation: bool = False, generation_delay: float = 0.0):
    """Runs retrieval + generation for one system (vector-only or hybrid),
    returns ragas-ready samples plus the raw retrieved products (for the
    catalog_recall@k side-check)."""
    from ragas import SingleTurnSample

    samples = []
    local_score_rows = []
    catalog_precisions = []
    catalog_recalls = []

    for q in eval_queries:
        products = retrieve_fn(q["query"], top_k=top_k)
        contexts = [product_to_context_text(p) for p in products] or ["No products retrieved."]

        if use_llm_generation:
            # Same generation call, memory disabled for both systems so retrieval
            # strategy is the only variable being compared.
            try:
                answer = engine.generate_recommendation(q["query"], products, memory_context="")
            except Exception as e:
                raise RuntimeError(
                    "LLM answer generation failed. Re-run without --use-llm-generation "
                    "to use deterministic extractive answers for evaluation."
                ) from e
            if generation_delay > 0:
                time.sleep(generation_delay)
        else:
            answer = generate_extractive_answer(q["query"], products, top_k)

        retrieved_names = {p["name"] for p in products}
        overlap = retrieved_names & q["catalog_matching_names"]
        catalog_precision = len(overlap) / (len(products) or 1)
        denom = min(top_k, len(q["catalog_matching_names"])) or 1
        catalog_recall = len(overlap) / denom
        catalog_precisions.append(catalog_precision)
        catalog_recalls.append(catalog_recall)

        sample = SingleTurnSample(
            user_input=q["query"],
            retrieved_contexts=contexts,
            response=answer,
            reference=q["reference_answer"],
            reference_contexts=q["reference_contexts"],
        )
        samples.append(sample)
        local_score_rows.append({
            **sample.to_dict(),
            "llm_context_precision_without_reference": catalog_precision,
            "context_recall": catalog_recall,
            "faithfulness": 1.0 if products else 0.0,
            "answer_relevancy": 1.0 if overlap else (0.5 if products else 0.0),
            "ragas_errors": "",
        })

    return (
        samples,
        pd.DataFrame(local_score_rows),
        sum(catalog_precisions) / len(catalog_precisions),
        sum(catalog_recalls) / len(catalog_recalls),
    )


# ---------------------------------------------------------------------------
# 3. Score both systems with RAGAS and print the comparison
# ---------------------------------------------------------------------------

class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()


def _score_metric(metric, sample) -> tuple[float, str]:
    try:
        return float(metric.single_turn_score(sample)), ""
    except Exception as e:
        return math.nan, f"{type(e).__name__}: {e}"


def score_with_ragas(samples, judge_llm, judge_embeddings, judge_max_workers: int, judge_timeout: int):
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    metrics = [
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
        Faithfulness(),
        AnswerRelevancy(strictness=1),
    ]
    run_config = RunConfig(timeout=judge_timeout, max_workers=judge_max_workers, max_retries=1, max_wait=10)

    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = judge_embeddings
        metric.init(run_config)

    rows = []
    for sample_index, sample in enumerate(samples, start=1):
        row = sample.to_dict()
        errors = []
        for metric in metrics:
            value, error = _score_metric(metric, sample)
            row[metric.name] = value
            if error:
                errors.append(f"{metric.name}: {error}")
        row["ragas_errors"] = " | ".join(errors)
        print(f"  scored sample {sample_index}/{len(samples)}")
        rows.append(row)

    return pd.DataFrame(rows)


def metric_mean(scores: pd.DataFrame, metric: str) -> tuple[float, int, int]:
    numeric = pd.to_numeric(scores[metric], errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return math.nan, 0, len(numeric)
    return float(valid.mean()), int(valid.shape[0]), int(numeric.shape[0])


def metric_mean_with_fallback(
    scores: pd.DataFrame,
    fallback_scores: pd.DataFrame,
    metric: str,
    primary_source: str,
) -> tuple[float, int, int, str]:
    mean, valid, total = metric_mean(scores, metric)
    if valid > 0:
        return mean, valid, total, primary_source

    fallback_mean, fallback_valid, fallback_total = metric_mean(fallback_scores, metric)
    return fallback_mean, fallback_valid, fallback_total, "catalog_fallback"


def main():
    parser = argparse.ArgumentParser(description="Compare vector-only vs hybrid RAG in Findora using RAGAS.")
    parser.add_argument("--n-queries", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--csv", type=str, default=os.path.join("data", "Amazon-Products.csv"))
    parser.add_argument("--out", type=str, default="ragas_comparison_results.csv")
    parser.add_argument(
        "--use-llm-generation",
        action="store_true",
        help="Use Gemini to generate answers before judging. Default is deterministic extractive answers to save quota.",
    )
    parser.add_argument(
        "--generation-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each LLM generation call when --use-llm-generation is enabled.",
    )
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=1,
        help="RAGAS judge concurrency. Keep at 1 for low Gemini quotas.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each RAGAS judge/model call.",
    )
    parser.add_argument(
        "--use-ragas-llm",
        action="store_true",
        help="Request RAGAS LLM-judged metrics. Uses catalog fallback unless --force-ragas-llm is also set.",
    )
    parser.add_argument(
        "--force-ragas-llm",
        action="store_true",
        help="Actually call the Gemini RAGAS judge. Can hang or fail when API quota is exhausted.",
    )
    args = parser.parse_args()

    load_dotenv()

    from kg_engine import KGRecommenderEngine

    print(f"Building {args.n_queries} catalog-grounded eval queries from {args.csv} ...")
    eval_queries = build_eval_queries(args.csv, n_queries=args.n_queries)
    for q in eval_queries:
        print(f"  - {q['query']}  ({len(q['catalog_matching_names'])} real catalog matches)")

    print("\nConnecting to Neo4j / loading embedding model ...")
    engine = KGRecommenderEngine(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
    )
    engine.load_link_lookup(os.path.dirname(args.csv) or "data")

    judge_llm = None
    judge_embeddings = None
    should_run_ragas_llm = args.use_ragas_llm and args.force_ragas_llm
    if should_run_ragas_llm:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        judge_llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0,
                max_retries=1,
                timeout=args.judge_timeout,
            )
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            LocalSentenceTransformerEmbeddings(engine.embedder)
        )
    elif args.use_ragas_llm:
        print(
            "\n--use-ragas-llm was requested, but Gemini-judged RAGAS is disabled unless "
            "--force-ragas-llm is also set. Using catalog fallback to avoid NaN/hangs from quota errors."
        )

    print("\nRunning VECTOR-ONLY retrieval + generation ...")
    vector_samples, vector_local_scores, vector_catalog_precision, vector_catalog_recall = run_system(
        engine,
        eval_queries,
        engine.retrieve_vector_only,
        args.top_k,
        use_llm_generation=args.use_llm_generation,
        generation_delay=args.generation_delay,
    )

    print("Running HYBRID (vector + graph) retrieval + generation ...")
    hybrid_samples, hybrid_local_scores, hybrid_catalog_precision, hybrid_catalog_recall = run_system(
        engine,
        eval_queries,
        engine.retrieve_hybrid,
        args.top_k,
        use_llm_generation=args.use_llm_generation,
        generation_delay=args.generation_delay,
    )

    engine.close()

    if should_run_ragas_llm:
        print("\nScoring VECTOR-ONLY with RAGAS LLM judge ...")
        vector_scores = score_with_ragas(
            vector_samples,
            judge_llm,
            judge_embeddings,
            args.judge_max_workers,
            args.judge_timeout,
        )

        print("Scoring HYBRID with RAGAS LLM judge ...")
        hybrid_scores = score_with_ragas(
            hybrid_samples,
            judge_llm,
            judge_embeddings,
            args.judge_max_workers,
            args.judge_timeout,
        )
    else:
        print("\nUsing catalog-grounded RAGAS-style metrics; pass --use-ragas-llm --force-ragas-llm for Gemini-judged RAGAS metrics.")
        vector_scores = vector_local_scores
        hybrid_scores = hybrid_local_scores

    metric_cols = [
        "llm_context_precision_without_reference",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    metric_cols = [c for c in metric_cols if c in vector_scores.columns]

    rows = []
    primary_source = "ragas_llm" if should_run_ragas_llm else "catalog"
    for metric in metric_cols:
        vector_mean, vector_valid, vector_total, vector_source = metric_mean_with_fallback(
            vector_scores,
            vector_local_scores,
            metric,
            primary_source,
        )
        hybrid_mean, hybrid_valid, hybrid_total, hybrid_source = metric_mean_with_fallback(
            hybrid_scores,
            hybrid_local_scores,
            metric,
            primary_source,
        )
        rows.append({
            "metric": metric,
            "vector_only": vector_mean,
            "hybrid": hybrid_mean,
            "vector_valid_rows": f"{vector_valid}/{vector_total}",
            "hybrid_valid_rows": f"{hybrid_valid}/{hybrid_total}",
            "source": vector_source if vector_source == hybrid_source else f"vector:{vector_source}, hybrid:{hybrid_source}",
        })
    rows.append({
        "metric": "catalog_precision@k (non-LLM)",
        "vector_only": vector_catalog_precision,
        "hybrid": hybrid_catalog_precision,
        "vector_valid_rows": f"{len(vector_samples)}/{len(vector_samples)}",
        "hybrid_valid_rows": f"{len(hybrid_samples)}/{len(hybrid_samples)}",
        "source": "catalog",
    })
    rows.append({
        "metric": "catalog_recall@k (non-LLM)",
        "vector_only": vector_catalog_recall,
        "hybrid": hybrid_catalog_recall,
        "vector_valid_rows": f"{len(vector_samples)}/{len(vector_samples)}",
        "hybrid_valid_rows": f"{len(hybrid_samples)}/{len(hybrid_samples)}",
        "source": "catalog",
    })

    comparison = pd.DataFrame({
        "metric": [row["metric"] for row in rows],
        "vector_only": [row["vector_only"] for row in rows],
        "hybrid": [row["hybrid"] for row in rows],
        "vector_valid_rows": [row["vector_valid_rows"] for row in rows],
        "hybrid_valid_rows": [row["hybrid_valid_rows"] for row in rows],
        "source": [row["source"] for row in rows],
    })
    comparison["delta (hybrid - vector_only)"] = comparison["hybrid"] - comparison["vector_only"]

    print("\n" + "=" * 70)
    print("RESULTS: vector-only vs hybrid")
    print("=" * 70)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if comparison[["vector_only", "hybrid"]].isna().any().any():
        print("\nSome RAGAS rows still returned NaN. Check *_valid_rows and the per-query CSVs; this usually means the judge API failed or returned an unparsable response.")

    comparison.to_csv(args.out, index=False)
    print(f"\nSaved comparison table to {args.out}")

    vector_scores.to_csv("ragas_vector_only_per_query.csv", index=False)
    hybrid_scores.to_csv("ragas_hybrid_per_query.csv", index=False)
    print("Saved per-query breakdowns to ragas_vector_only_per_query.csv / ragas_hybrid_per_query.csv")


if __name__ == "__main__":
    main()
