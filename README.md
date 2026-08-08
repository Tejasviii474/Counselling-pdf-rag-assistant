# Counselling PDF Assistant

A general-purpose RAG chat assistant for exam counselling brochures — upload any
counselling PDF, ask questions in a ChatGPT-style interface, get grounded answers
with page citations, instead of reading 20-50 pages to find one detail. Built after
personally struggling to parse the CCMN 2025 (IIT JAM) counselling brochure.

**Scope note**: the pipeline is exam-agnostic by design (it doesn't hardcode
anything about JAM specifically). It was built and evaluated in depth against the
CCMN 2025 (IIT JAM) brochure (see Evaluation below), and additionally spot-tested
against the JoSAA 2026 (JEE) Business Rules document to check generalization to a
second, differently-formatted exam brochure. Spot-testing surfaced both a
confirmation (definitional questions like Freeze/Float/Slide answered correctly
with accurate citations) and a new, distinct limitation around complex table
extraction -- see Known limitations.

## Why this exists

Counselling brochures mix dense procedural text, fee tables, category rules, and
annexures — they're genuinely hard to parse correctly, and painful to read under
time pressure during an actual admissions window. This project solves both: layout-
aware extraction on the engineering side, instant grounded answers on the user side.

## Architecture

```
PDF upload (Streamlit)
        │
        ▼
Parsing (pdfplumber -- text + tables)
        │
        ▼
Chunking (section + table aware, ~1000 char target)
        │
        ▼
Embedding (local HuggingFace bge-small-en-v1.5) + Chroma vector store
        │
        ▼
Hybrid retrieval (BM25 keyword search + vector search, reciprocal rank fusion)
        │
        ▼
Generation (Gemini, grounded-only prompt with page citations)
        │
        ▼
Streamlit chat interface
```

Everything is built fresh per uploaded PDF and held in session state — no
persistent database, so any counselling PDF can be dropped in and queried
immediately.

## Tech stack

- **LangChain** for the retrieval/generation pipeline
- **pdfplumber** for layout-aware PDF parsing (text + tables extracted separately)
- **Chroma** (in-memory) for vector storage
- **HuggingFace `bge-small-en-v1.5`** for embeddings (local, free, no API cost)
- **`rank_bm25`** for keyword-based retrieval, combined with vector search via
  reciprocal rank fusion
- **Google Gemini API** (free tier) for answer generation
- **Streamlit** for the chat interface

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-key-here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

Run the app:
```bash
streamlit run app.py
```

Upload a counselling PDF and start asking questions.

## Evaluation

I built a 15-question test set against the actual CCMN 2025 (IIT JAM) brochure,
with ground truth manually verified against the source document rather than
guessed — including one deliberate hallucination-guard test (asking about "CCMT",
a term that does not appear anywhere in the brochure).

### Baseline (vector-only retrieval)

7 of 15 questions passed an automated ground-truth check. Manual review showed the
real number was closer to **11/14 scored questions substantively correct** — several
"failures" were false negatives from strict phrase-matching (e.g. expecting the word
"three" when the model correctly wrote "3"), not actual content errors.

Genuine retrieval misses at baseline: 3 questions where the correct answer existed
in the document but wasn't retrieved, because vector search favored topically-similar
chunks over the one chunk containing the specific fact (a fee amount, a deadline
date, a refund rule).

### After hybrid retrieval (BM25 + vector, reciprocal rank fusion)

Added keyword-based retrieval alongside vector search to catch exact terms (fee
amounts, specific dates, category codes) that pure semantic search sometimes misses.

- **Fixed**: the fee-lookup failure (National Spot Round fee) — BM25 correctly
  surfaced the fee table chunk that vector search consistently ranked too low.
- **Side effect observed**: fixing one retrieval path shifted ranking elsewhere,
  causing a different, previously-passing question to retrieve a less complete
  answer. This is a real, common RAG tradeoff, not a regression I ignored — noted
  here rather than hidden.

### After increasing retrieval depth (k=4 → k=6)

- **Fixed**: a definitional question (Float/Slide/Freeze) that needed content
  spanning two separate sections of the document — more retrieval slots let both
  relevant chunks through.
- **Did not fix**: 2 remaining questions (a provisional-admission eligibility
  question, and a refund-amount question) where the correct chunk never ranked
  competitively in *either* retrieval method, regardless of how many results were
  requested. This indicates the issue is chunk-level (how that content was split
  and embedded), not retrieval-depth. Documented as a known limitation rather than
  chased further, given diminishing returns.

### Final result

9/14 scored questions passing an automated check; qualitative review puts the real
number higher, since several "failures" are exact-phrase-matching artifacts rather
than content errors. The hallucination-guard question was correctly refused in
every single run across all retrieval configurations tested.

## Known limitations

- **Complex multi-column/merged-cell schedule tables are not reliably captured.**
  Spot-testing on the JoSAA 2026 brochure found that a detailed round-by-round
  schedule table (dates, deadlines, and conditional NIT+-specific rows across
  merged cells) produced an answer with only a single fact ("5 rounds, Round 5 is
  final") when the source table actually contained a full multi-week schedule with
  per-round deadlines. This is a distinct failure mode from the CCMN retrieval
  misses below -- here the extraction step itself (`pdfplumber`'s table detection)
  appears to lose structure on complex nested tables, rather than the correct
  chunk simply ranking too low in retrieval. Simpler tables (flat fee structures,
  document checklists) extracted reliably on both documents tested. A more robust
  fix would likely involve `unstructured`'s layout-model-based table extraction
  instead of `pdfplumber`, which was considered during design but not used in this
  build.
- Two questions (provisional admission deadlines, specific refund amounts) reliably
  fail to retrieve the correct chunk under the current chunking strategy. Root
  cause: the relevant text is chunked in a way that doesn't rank competitively for
  the natural phrasing of those questions in either BM25 or vector search. A fix
  would likely require revisiting chunk boundaries around dense procedural/numeric
  content, not just retrieval-side tuning.
- The automated eval's `expected_answer_contains` phrase-matching is strict
  (case-sensitive-adjacent, exact-string) and produces false negatives on answers
  that are substantively correct but phrased differently than the expected keyword
  (e.g. "3" vs "three", "cannot" vs "not be able to"). A production version of this
  eval would use fuzzy matching or LLM-graded correctness instead.
- Free-tier Gemini API rate limits (5 requests/minute) require pacing in the eval
  script; this is a constraint of the free tier, not the pipeline itself.

## Extending

- Test against additional exam brochures (GATE/COAP, UGC NET) to further confirm
  generalization beyond the two exams (JAM, JEE) already spot-tested.
- Swap `pdfplumber` table extraction for `unstructured`'s layout-model-based
  extraction to address the complex-table limitation found during JoSAA testing.
- Revisit chunk boundaries for dense tables/numeric sections to address the two
  known retrieval misses above.
- Add conversational memory for follow-up questions within the same chat session.
