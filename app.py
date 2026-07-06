"""
Counselling PDF assistant — full pipeline wired in.
Upload a PDF -> parse -> chunk -> embed -> ask questions in chat.
"""
import streamlit as st
from dotenv import load_dotenv 
load_dotenv()


from src.parse_docs import parse_pdf
from src.chunk_docs import chunk_elements
from src.build_index import build_vector_store
from src.hybrid_retrieve import HybridRetriever
from src.generate import answer_question

st.set_page_config(page_title="Counselling PDF Assistant", page_icon="📄")
st.title("Counselling PDF Assistant")
st.caption("Upload a counselling brochure PDF and ask questions about it.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload counselling PDF", type=["pdf"])

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

# --- Build the index only when a NEW file is uploaded (not on every rerun) ---
if uploaded_file is not None:
    if st.session_state.processed_filename != uploaded_file.name:
        with st.spinner(f"Reading and indexing {uploaded_file.name}... (first time only, ~30s)"):
            elements = parse_pdf(uploaded_file)
            chunks = chunk_elements(elements)
            vectordb = build_vector_store(chunks)
            st.session_state.retriever = HybridRetriever(vectordb, chunks)
            st.session_state.processed_filename = uploaded_file.name
            st.session_state.messages = []  # reset chat when a new PDF is loaded
        st.success(f"Ready! Loaded {len(chunks)} chunks from {uploaded_file.name}")
    else:
        st.success(f"Using: {uploaded_file.name}")
else:
    st.info("Upload a PDF to get started.")

# --- Render existing chat messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
query = st.chat_input("Ask a question about the uploaded PDF...")

if query:
    if st.session_state.retriever is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = st.session_state.retriever.hybrid_search(query, k=6)
                answer = answer_question(query, retrieved)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})