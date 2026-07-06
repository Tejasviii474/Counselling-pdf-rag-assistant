from src.parse_docs import parse_pdf
from src.chunk_docs import chunk_elements
from src.build_index import build_vector_store
from src.hybrid_retrieve import HybridRetriever

with open("data/2025051422.pdf", "rb") as f:
    elements = parse_pdf(f)
chunks = chunk_elements(elements)
vectordb = build_vector_store(chunks)

retriever = HybridRetriever(vectordb, chunks)

query = "What is the fee for the National Spot Round?"
print("--- Vector-only results ---")
for text, meta in retriever.vector_search(query, k=4):
    print(f"Pages {meta['page_numbers']}: {text[:100]}")

print("\n--- Hybrid results ---")
for text, meta in retriever.hybrid_search(query, k=4):
    print(f"Pages {meta['page_numbers']}: {text[:100]}")