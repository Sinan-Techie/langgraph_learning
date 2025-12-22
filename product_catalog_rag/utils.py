import httpx
import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

#function to call LLM

def call_llm(prompt:str) -> str:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1000,
        api_key=api_key
    )

    response = llm.invoke([prompt])
    print("LLM response object:", response)
    return response.content 

def pretty_print_batch_data(batch_data: list[dict]) -> None:
    print("\n================ VECTOR SEARCH RESULTS ================\n")

    for idx, item in enumerate(batch_data, start=1):
        print(f"Query {idx}: {item['query']}")
        print("-" * 60)

        if not item["candidates"]:
            print("  No candidates found.")
        else:
            for c_idx, cand in enumerate(item["candidates"], start=1):
                print(f"  Candidate {c_idx}:")
                print(f"    Product ID   : {cand['product_id']}")
                print(f"    Product Name : {cand['product_name']}")
                print(f"    Category     : {cand['category']}")
                print(f"    Distance     : {cand['distance']}")
                print(f"    Description  : {cand['description'][:200]}...")
                print()

        print("=" * 60)

    print("\n=======================================================\n")

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def hybrid_retrieve(query, collection, bm25, documents, metadatas, ids, top_k=10):
    # ---------------------------
    # VECTOR SEARCH (Recall)
    # ---------------------------
    vector_results = collection.query(
        query_texts=[query],
        n_results=25,  # high recall
        include=["documents", "metadatas", "distances"]
    )

    vector_candidates = {}
    for doc, meta, dist in zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0]
    ):
        pid = meta["product_id"]
        vector_candidates[pid] = {
            "product_id": pid,
            "product_name": meta["product_name"],
            "category": meta["category"],
            "distance": dist,
            "doc": doc,
            "bm25": 0.0,
            "numeric_match": 0
        }

    # ---------------------------
    # KEYWORD SEARCH (BM25)
    # ---------------------------
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)

    for idx, score in enumerate(bm25_scores):
        if score <= 0:
            continue

        pid = metadatas[idx]["product_id"]

        if pid not in vector_candidates:
            vector_candidates[pid] = {
                "product_id": pid,
                "product_name": metadatas[idx]["product_name"],
                "category": metadatas[idx]["category"],
                "distance": 1.0,  # worst distance
                "doc": documents[idx],
                "bm25": score,
                "numeric_match": 0
            }
        else:
            vector_candidates[pid]["bm25"] = score

    # ---------------------------
    # NUMERIC MATCH BOOST
    # ---------------------------
    numbers_in_query = set(re.findall(r"\d+", query))

    for c in vector_candidates.values():
        numbers_in_name = set(re.findall(r"\d+", c["product_name"]))
        if numbers_in_query & numbers_in_name:
            c["numeric_match"] = 1

    # ---------------------------
    # NORMALIZATION
    # ---------------------------
    distances = [1 - c["distance"] for c in vector_candidates.values()]
    bm25s = [c["bm25"] for c in vector_candidates.values()]

    def normalize(xs):
        if not xs or max(xs) == min(xs):
            return xs
        return [(x - min(xs)) / (max(xs) - min(xs)) for x in xs]

    norm_vec = normalize(distances)
    norm_bm25 = normalize(bm25s)

    # ---------------------------
    # FINAL HYBRID SCORE
    # ---------------------------
    ALPHA = 0.5   # semantic
    BETA = 0.3    # keyword
    GAMMA = 0.2   # numeric identity

    for c, v, b in zip(vector_candidates.values(), norm_vec, norm_bm25):
        c["hybrid_score"] = (
            ALPHA * v +
            BETA * b +
            GAMMA * c["numeric_match"]
        )

    # ---------------------------
    # SORT & RETURN
    # ---------------------------
    return sorted(
        vector_candidates.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )[:top_k]
