"""
Step 3 — Chunking
Takes the elements from parse_docs.py and groups them into retrievable chunks.
Tables always become their own standalone chunk (never split mid-table).
Text elements get merged until they reach a target size, then split.
"""

CHUNK_TARGET_CHARS = 1000


def chunk_elements(elements: list[dict]) -> list[dict]:
    chunks = []
    buffer_text = []
    buffer_pages = []

    def flush_buffer():
        if buffer_text:
            chunks.append(
                {
                    "text": "\n".join(buffer_text).strip(),
                    "page_numbers": buffer_pages.copy(),
                    "chunk_type": "text",
                }
            )
        buffer_text.clear()
        buffer_pages.clear()

    for el in elements:
        if el["type"] == "table":
            flush_buffer()
            chunks.append(
                {
                    "text": el["text"],
                    "page_numbers": [el["page_number"]],
                    "chunk_type": "table",
                }
            )
            continue

        buffer_text.append(el["text"])
        buffer_pages.append(el["page_number"])

        current_len = sum(len(t) for t in buffer_text)
        if current_len > CHUNK_TARGET_CHARS:
            flush_buffer()

    flush_buffer()
    return [c for c in chunks if c["text"].strip()]


if __name__ == "__main__":
    import sys
    from src.parse_docs import parse_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m src.chunk_docs <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        elements = parse_pdf(f)

    chunks = chunk_elements(elements)
    print(f"Created {len(chunks)} chunks from {len(elements)} elements\n")

    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    print(f"({len(table_chunks)} of them are table chunks)\n")

    for c in chunks[:3]:
        print(f"--- Pages {c['page_numbers']} ({c['chunk_type']}) ---")
        print(c["text"][:300])
        print()