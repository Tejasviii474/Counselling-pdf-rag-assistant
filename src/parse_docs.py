"""
Step 2 — Parsing
Extracts text and tables from an uploaded PDF using pdfplumber.
Returns a list of "elements" (text blocks or tables) tagged with page numbers,
which chunk_docs.py will group into retrievable chunks in the next step.

We start with pdfplumber only (simpler, no extra system dependencies like
poppler/tesseract). We can upgrade to `unstructured` later once the basic
pipeline works end-to-end.
"""
import pdfplumber


def parse_pdf(file_obj) -> list[dict]:
    """
    file_obj: a file-like object (e.g. Streamlit's UploadedFile, or an open file)
    Returns: list of dicts like {"page_number": int, "type": "text"|"table", "text": str}
    """
    elements = []

    with pdfplumber.open(file_obj) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables first, so we can exclude their text from the
            # plain text extraction below (avoids duplicating table content)
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                rows_text = "\n".join(
                    " | ".join(cell or "" for cell in row) for row in table
                )
                elements.append(
                    {
                        "page_number": page_num,
                        "type": "table",
                        "text": f"Table {t_idx + 1} (page {page_num}):\n{rows_text}",
                    }
                )

            # Extract plain text
            text = page.extract_text()
            if text and text.strip():
                elements.append(
                    {
                        "page_number": page_num,
                        "type": "text",
                        "text": text.strip(),
                    }
                )

    return elements


# Quick manual test when run directly: python -m src.parse_docs <path_to_pdf>
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.parse_docs <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        result = parse_pdf(f)

    print(f"Extracted {len(result)} elements from {path}")
    for el in result[:3]:
        print(f"\n--- Page {el['page_number']} ({el['type']}) ---")
        print(el["text"][:300])