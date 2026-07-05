# Turn retrieved chunks into a grounded answer

"""
Step 5 — Generation
Takes retrieved chunks + the user's question, and produces a grounded answer
that cites page numbers and refuses to guess when the answer isn't in the chunks.
Uses Google's Gemini API (free tier).
"""
import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

SYSTEM_PROMPT = """You are a counselling assistant that helps students understand
official exam counselling brochures.

Rules:
- Answer ONLY using the provided context chunks below. Do not use outside knowledge.
- If the answer is not present in the context, say clearly: "I couldn't find this in
  the brochure — please check the official notification directly."
- Always cite the page number(s) your answer is based on, like: (Source: page 12).
- Be concise and direct.
"""


def format_context(retrieved_docs) -> str:
    blocks = []
    for text, metadata in retrieved_docs:
        pages = metadata.get("page_numbers", "unknown")
        blocks.append(f"[Pages {pages}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(query: str, retrieved_docs) -> str:
    context = format_context(retrieved_docs)
    prompt = f"""Context from the brochure:

{context}

---

Student's question: {query}

Answer using only the context above, and cite page numbers."""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text


if __name__ == "__main__":
    import sys
    from src.parse_docs import parse_pdf
    from src.chunk_docs import chunk_elements
    from src.build_index import build_vector_store

    if len(sys.argv) < 3:
        print('Usage: python -m src.generate <path_to_pdf> "<question>"')
        sys.exit(1)

    path = sys.argv[1]
    question = sys.argv[2]

    with open(path, "rb") as f:
        elements = parse_pdf(f)
    chunks = chunk_elements(elements)
    vectordb = build_vector_store(chunks)

    from src.hybrid_retrieve import HybridRetriever
    retriever = HybridRetriever(vectordb, chunks)
    retrieved = retriever.hybrid_search(question, k=4)
    answer = answer_question(question, retrieved)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}")