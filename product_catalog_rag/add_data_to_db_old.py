import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# -----------------------------
# CONFIG
# -----------------------------
EXCEL_PATH = "data/product_catalog.xlsx"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "products_catalog"

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
# PERSISTENT Chroma client (IMPORTANT)
# -----------------------------
client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

# Recreate collection (safe re-index)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# -----------------------------
# Load Excel
# -----------------------------
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# -----------------------------
# Build documents
# -----------------------------
documents = []
ids = []
metadatas = []

for _, row in df.iterrows():
    doc_text = f"""
    Product Name: {row['Product_Name']}
    Description: {row['Product_Description']}
    Category: {row['Category']}
    Sub Category: {row['Sub_Category']}
    Brand: {row['Brand']}
    Industry Use: {row['Industry_Use']}
    Form Factor: {row['Form_Factor']}
    Interface: {row['Interface_Type']}
    """

    documents.append(doc_text.strip())
    ids.append(str(row["Product_ID"]))
    metadatas.append({
        "product_id": row["Product_ID"],
        "product_name": row["Product_Name"],
        "brand": row["Brand"],
        "category": row["Category"],
        "status": row["Lifecycle_Status"]
    })

# -----------------------------
# Add to vector DB
# -----------------------------
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Indexed {len(ids)} products into persistent Chroma DB.")
