import streamlit as st
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "products_catalog"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="ChromaDB Viewer",
    layout="wide"
)

st.title("🔍 ChromaDB Collection Viewer")
st.caption("Inspect documents, metadata, and embeddings stored in ChromaDB")

# -------------------------------------------------
# INIT CHROMA CLIENT
# -------------------------------------------------
@st.cache_resource
def load_collection():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return collection

collection = load_collection()

# -------------------------------------------------
# COLLECTION STATS
# -------------------------------------------------
count = collection.count()

st.metric("Total Records in Collection", count)

# -------------------------------------------------
# FETCH DATA
# -------------------------------------------------
with st.spinner("Loading records from ChromaDB..."):
    data = collection.get(
        include=["documents", "metadatas"]  # embeddings optional
    )

# -------------------------------------------------
# PREPARE TABLE
# -------------------------------------------------
rows = []

for idx, doc in enumerate(data["documents"]):
    meta = data["metadatas"][idx]

    rows.append({
        "Product ID": meta.get("product_id"),
        "Product Name": meta.get("product_name"),
        "Brand": meta.get("brand"),
        "Category": meta.get("category"),
        "Status": meta.get("status"),
        "Document Preview": doc[:300] + "..."
    })

df = pd.DataFrame(rows)

# -------------------------------------------------
# FILTERS
# -------------------------------------------------
st.subheader("Filters")

col1, col2 = st.columns(2)

with col1:
    category_filter = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(df["Category"].dropna().unique().tolist())
    )

with col2:
    brand_filter = st.selectbox(
        "Filter by Brand",
        ["All"] + sorted(df["Brand"].dropna().unique().tolist())
    )

filtered_df = df.copy()

if category_filter != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category_filter]

if brand_filter != "All":
    filtered_df = filtered_df[filtered_df["Brand"] == brand_filter]

# -------------------------------------------------
# DISPLAY TABLE
# -------------------------------------------------
st.subheader("Indexed Products")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=600
)

# -------------------------------------------------
# RAW RECORD VIEWER
# -------------------------------------------------
st.subheader("🔎 Inspect Raw Record")

selected_row = st.number_input(
    "Enter row index to inspect",
    min_value=0,
    max_value=len(data["documents"]) - 1,
    step=1
)

st.markdown("### Document")
st.code(data["documents"][selected_row], language="text")

st.markdown("### Metadata")
st.json(data["metadatas"][selected_row])

# -------------------------------------------------
# OPTIONAL: EMBEDDINGS VIEW
# -------------------------------------------------
with st.expander("⚠️ View Embedding Vector (Advanced)"):
    emb_data = collection.get(
        include=["embeddings"],
        ids=[collection.get()["ids"][selected_row]]
    )
    st.write(f"Embedding dimension: {len(emb_data['embeddings'][0])}")
    st.write(emb_data["embeddings"][0][:20], "...")

