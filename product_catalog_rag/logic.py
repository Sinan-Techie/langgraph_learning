import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import re
import json
import uuid

from prompt_builder import (
    build_llm_prompt_batch,
    build_input_normalization_prompt
)
from utils import call_llm, pretty_print_batch_data

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "products_catalog"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
TOP_K = 10
DEBUG_HYBRID = True

# -----------------------------
# Embedding function
# -----------------------------
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# Chroma client
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# -----------------------------
# LOAD ALL DOCUMENTS (for BM25)
# -----------------------------
def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

all_data = collection.get(include=["documents", "metadatas"])

documents = all_data["documents"]
metadatas = all_data["metadatas"]

bm25_corpus = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(bm25_corpus)

# =====================================================
# HYBRID RETRIEVAL
# =====================================================
def hybrid_retrieve(query: str, top_k: int = 10):
    # ---- Vector recall ----
    vector_results = collection.query(
        query_texts=[query],
        n_results=25,
        include=["documents", "metadatas", "distances"]
    )

    candidates = {}

    for doc, meta, dist in zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0]
    ):
        pid = meta["product_id"]
        candidates[pid] = {
            "product_id": pid,
            "product_name": meta["product_name"],
            "category": meta["category"],
            "doc": doc,
            "distance": dist,
            "bm25": 0.0,
            "numeric_match": 0
        }

    # ---- BM25 keyword ----
    scores = bm25.get_scores(tokenize(query))
    for idx, score in enumerate(scores):
        if score <= 0:
            continue

        pid = metadatas[idx]["product_id"]
        if pid not in candidates:
            candidates[pid] = {
                "product_id": pid,
                "product_name": metadatas[idx]["product_name"],
                "category": metadatas[idx]["category"],
                "doc": documents[idx],
                "distance": 1.0,
                "bm25": score,
                "numeric_match": 0
            }
        else:
            candidates[pid]["bm25"] = score

    # ---- Numeric identity boost ----
    q_nums = set(re.findall(r"\d+", query))
    for c in candidates.values():
        name_nums = set(re.findall(r"\d+", c["product_name"]))
        if q_nums & name_nums:
            c["numeric_match"] = 1
    

    if DEBUG_HYBRID:
        print("\n================ HYBRID DEBUG ================")
        print(f"QUERY: {query}\n")

        for c in candidates.values():
            if c['product_name'] in ["Dynasty Max Server Components 345", "Spectra Max Server Components 218"]:
                print(f"Product: {c['product_name']}")
                print(f"  Distance        : {c['distance']:.4f}")
                print(f"  Semantic score  : {1 - c['distance']:.4f}")
                print(f"  BM25 score      : {c['bm25']:.4f}")
                print(f"  Numeric match   : {c['numeric_match']}")
                print()

    # ---- Normalize & score ----
    vec_scores = [1 - c["distance"] for c in candidates.values()]
    bm25_scores = [c["bm25"] for c in candidates.values()]

    def norm(xs):
        if not xs or max(xs) == min(xs):
            return xs
        return [(x - min(xs)) / (max(xs) - min(xs)) for x in xs]

    n_vec = norm(vec_scores)
    n_bm25 = norm(bm25_scores)

    if DEBUG_HYBRID:
        print("---- Normalized Scores ----")
        for c, v, b in zip(candidates.values(), n_vec, n_bm25):
            if c['product_name'] in ["Dynasty Max Server Components 345", "Spectra Max Server Components 218"]:
                print(f"{c['product_name']}")
                print(f"  Normalized semantic : {v:.4f}")
                print(f"  Normalized BM25     : {b:.4f}")
                print(f"  Numeric match       : {c['numeric_match']}")
                print()


    for c, v, b in zip(candidates.values(), n_vec, n_bm25):
        semantic_part = 0.5 * v
        keyword_part = 0.3 * b
        numeric_part = 0.2 * c["numeric_match"]

        c["hybrid_score"] = semantic_part + keyword_part + numeric_part

        if DEBUG_HYBRID:
            # if c['product_name'] in ["Dynasty Max Server Components 345", "Spectra Max Server Components 218"]:
            print(f"FINAL SCORE → {c['product_name']}")
            print(f"  Semantic part (0.5 * {v:.4f}) = {semantic_part:.4f}")
            print(f"  Keyword part  (0.3 * {b:.4f}) = {keyword_part:.4f}")
            print(f"  Numeric part  (0.2 * {c['numeric_match']}) = {numeric_part:.4f}")
            print(f"  HYBRID SCORE  = {c['hybrid_score']:.4f}")
            print("--------------------------------------------")

    return sorted(
        candidates.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )[:top_k]

# =====================================================
# STAGE 0: INPUT NORMALIZATION
# =====================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_input_text = f.read().strip()

normalization_prompt = build_input_normalization_prompt(raw_input_text)
normalized_result = call_llm(normalization_prompt)

queries = [q.strip() for q in normalized_result.splitlines() if q.strip()]

# =====================================================
# STAGE 1: HYBRID RETRIEVAL
# =====================================================
batch_data = []

for query in queries:
    results = hybrid_retrieve(query, TOP_K)

    batch_data.append({
        "query": query,
        "candidates": [
            {
                "product_id": c["product_id"],
                "product_name": c["product_name"],
                "category": c["category"],
                "distance": round(c["distance"], 4),
                "hybrid_score": round(c["hybrid_score"], 4),
                "description": c["doc"][:300]
            }
            for c in results
        ]
    })

pretty_print_batch_data(batch_data)

# =====================================================
# STAGE 2: LLM CANONICAL SELECTION
# =====================================================
selection_prompt = build_llm_prompt_batch(batch_data)
llm_result = call_llm(selection_prompt)

final_outputs = json.loads(llm_result)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_outputs, f, indent=2)

print("LLM selection completed successfully.")
