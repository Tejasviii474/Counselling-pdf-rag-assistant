#embed chunks and build a searchable vector store

"""
Step 4 — Embedding and indexing
Embeds each chunk using a local HuggingFace model (free, no API key needed)
and stores them in an in-memory Chroma vector store for similarity search.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_vector_store(chunks: list[dict]):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "page_numbers": str(c["page_numbers"]),
                "chunk_type": c["chunk_type"],
            },
        )
        for c in chunks
    ]

    # in-memory only for now — no persist_directory — rebuilt fresh each
    # time a new PDF is uploaded in the app
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectordb


if __name__ == "__main__":
    import sys
    from src.parse_docs import parse_pdf
    from src.chunk_docs import chunk_elements

    if len(sys.argv) < 2:
        print("Usage: python -m src.build_index <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        elements = parse_pdf(f)
    chunks = chunk_elements(elements)

    print(f"Embedding {len(chunks)} chunks... (this may take a minute on first run)")
    vectordb = build_vector_store(chunks)
    print("Vector store built successfully.")

    # quick manual test: search for something and see what comes back
    test_query = "what is the registration process"
    results = vectordb.similarity_search(test_query, k=2)
    print(f"\nTest query: '{test_query}'")
    for r in results:
        print(f"\n--- Match (pages {r.metadata['page_numbers']}) ---")
        print(r.page_content[:300])