# app.py
import streamlit as st
from main import ingest_pdfs, load_pipeline, ask_with_pipeline, reset_chroma

st.set_page_config(page_title="📄 PDF QA Agent", layout="wide")

# --- Sidebar controls ---
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model", ["small", "base", "large"], index=0)
topk = st.sidebar.slider("Top-K Chunks", 1, 10, 5)
if st.sidebar.button("Reset ChromaDB"):
    reset_chroma()
    st.sidebar.success("Chroma store has been reset!")

# --- File uploader for ingestion ---
st.title("📄 PDF Question Answering Agent")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded in uploaded_files:
        # Save uploaded PDFs locally
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getbuffer())
        ingest_pdfs(uploaded.name)
    st.success(f"Ingested {len(uploaded_files)} file(s) into ChromaDB.")

# --- Ask questions ---
st.subheader("💬 Ask a Question")
question = st.text_input("Enter your question here:")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = load_pipeline(model_alias=model_choice)

if st.button("Ask") and question:
    with st.spinner("Generating answer..."):
        result, retrieved = ask_with_pipeline(
            st.session_state.pipeline, question, top_k=topk, return_retrieved=True
        )

    st.markdown("### ✅ Answer")
    st.write(result["answer"])

    # Citations
    if result["citations"]:
        st.markdown("### 📚 Citations")
        for c in result["citations"]:
            st.write(f"- {c['filename']} p.{c['page']}")
    else:
        st.info("No citations found.")

    # Retrieved chunks
    with st.expander("🔎 Retrieved Chunks"):
        for r in retrieved:
            fn = r['metadata'].get('filename')
            pg = r['metadata'].get('page_number')
            txt = r['document'][:500].replace("\n", " ").strip()
            st.markdown(f"**{fn} (p.{pg})**: {txt}...")

    st.caption(f"⏱ Response generated in {result['elapsed']:.2f} seconds")
