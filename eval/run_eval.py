"""
Runs all test questions through the RAG pipeline and prints results
for manual verification against the actual PDF. Once you've confirmed
answers are correct, fill in expected_answer_contains / expected_page_range
in test_questions.json, then re-run this to get automated pass/fail scoring.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.parse_docs import parse_pdf
from src.chunk_docs import chunk_elements
from src.build_index import build_vector_store
from src.hybrid_retrieve import HybridRetriever
from src.generate import answer_question


def load_questions():
    path = Path(__file__).resolve().parent / "test_questions.json"
    return json.loads(path.read_text())


def has_ground_truth(q: dict) -> bool:
    return bool(q["expected_answer_contains"]) and q["expected_page_range"][0] is not None


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_eval <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    questions = load_questions()

    print(f"Building index from {pdf_path}...\n")
    with open(pdf_path, "rb") as f:
        elements = parse_pdf(f)
    chunks = chunk_elements(elements)
    vectordb = build_vector_store(chunks)
    retriever = HybridRetriever(vectordb, chunks)

    results = []
    for q in questions:
        import time
        retrieved = retriever.hybrid_search(q["question"], k=6)

        for attempt in range(4):
            try:
                answer = answer_question(q["question"], retrieved)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s, 240s
                    print(f"  Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/4)")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            answer = "ERROR: failed after 4 retries"

        time.sleep(15)  # ~4 requests/minute, safely under the 5/min limit  # stay under 5 requests/minute even without hitting the limit
        retrieved_pages = [meta.get("page_numbers") for _, meta in retrieved]

        print(f"\n{'='*70}")
        print(f"Q{q['id']}: {q['question']}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Answer:\n{answer}")

        result = {"id": q["id"], "question": q["question"], "answer": answer,
                   "retrieved_pages": retrieved_pages}

        if has_ground_truth(q):
            answer_lower = answer.lower()
            contains_check = all(
                phrase.lower() in answer_lower for phrase in q["expected_answer_contains"]
            )
            result["contains_check_passed"] = contains_check
            print(f"Ground-truth phrase check: {'PASS' if contains_check else 'FAIL'}")
        else:
            print("(No ground truth -- likely the hallucination-guard test question)")

        results.append(result)

    out_path = Path(__file__).resolve().parent / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    scored = [r for r in results if "contains_check_passed" in r]
    if scored:
        passed = sum(1 for r in scored if r["contains_check_passed"])
        print(f"\n\n{'='*70}")
        print(f"SCORED: {passed}/{len(scored)} passed (out of {len(questions)} total questions)")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()