import chromadb
from chromadb.utils import embedding_functions
from prompt_builder import (
    build_llm_prompt_batch,
    build_input_normalization_prompt
)
from utils import call_llm,pretty_print_batch_data
import json
import uuid

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "products_catalog"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
TOP_K = 10

# -----------------------------
# Embedding function
# -----------------------------
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
# embedding_function = embedding_functions.OpenAIEmbeddingFunction(
#     api_key="",  # or omit if env var is set
#     model_name="text-embedding-3-large"
# )
# -----------------------------
# Persistent Chroma client
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

trace_id = str(uuid.uuid4())

# =====================================================
# STAGE 0: INPUT NORMALIZATION (LLM)
# =====================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_input_text = f.read().strip()

print("Raw input text:\n", raw_input_text)

normalization_prompt = build_input_normalization_prompt(raw_input_text)

normalized_result = call_llm(normalization_prompt)
print("Normalized LLM result:", normalized_result)
# Normalize return type
if isinstance(normalized_result, str):
    queries = [
        line.strip()
        for line in normalized_result.splitlines()
        if line.strip()
    ]
elif isinstance(normalized_result, list):
    queries = normalized_result
elif isinstance(normalized_result, dict) and "queries" in normalized_result:
    queries = normalized_result["queries"]
else:
    raise ValueError("Failed to extract queries from input")

if not queries:
    raise ValueError("No valid queries extracted from input")

# print("Normalized queries:")
for q in queries:
    print("-", q)

# =====================================================
# STAGE 1: VECTOR SEARCH (PER QUERY)
# =====================================================
batch_data = []

for query in queries:
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        candidates.append({
            "product_id": meta["product_id"],
            "product_name": meta.get("product_name", meta["product_id"]),
            "category": meta["category"],
            "distance": round(dist, 4),
            "description": doc[:500]
        })

    batch_data.append({
        "query": query,
        "candidates": candidates
    })

# print("Vector search completed for all queries.")
pretty_print_batch_data(batch_data)


# =====================================================
# STAGE 2: SINGLE LLM CANONICAL SELECTION
# =====================================================
selection_prompt = build_llm_prompt_batch(batch_data)

llm_result = call_llm(selection_prompt)

# print("LLM selection raw result:", llm_result)

# Normalize LLM output
if isinstance(llm_result, list):
    final_outputs = llm_result

elif isinstance(llm_result, str):
    try:
        final_outputs = json.loads(llm_result)
    except json.JSONDecodeError:
        final_outputs = []

elif isinstance(llm_result, dict) and "response" in llm_result:
    final_outputs = llm_result["response"]

else:
    final_outputs = [
        {
            "input_query": item["query"],
            "selected_product_id": None,
            "selected_product_name": None,
            "confidence": "low",
            "reason": "LLM returned no usable response"
        }
        for item in batch_data
    ]

# Safety check
if len(final_outputs) != len(batch_data):
    raise ValueError(
        f"LLM returned {len(final_outputs)} outputs for "
        f"{len(batch_data)} queries"
    )

# -----------------------------
# Write output
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(final_outputs, out, indent=2)

print("LLM selection completed successfully.")
