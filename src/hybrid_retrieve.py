"""
Hybrid retrieval: combines vector search (semantic) with BM25 (exact keyword
matching). Fixes cases where vector search finds topically-similar-but-wrong
chunks over the one chunk containing the exact fact/number needed.
"""
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vectordb, chunks: list[dict]):
        """
        vectordb: the Chroma vector store (already built)
        chunks: the same list of chunk dicts used to build vectordb,
                needed here to build a matching BM25 index
        """
        self.vectordb = vectordb
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, k: int = 5):
        results = self.vectordb.similarity_search(query, k=k)
        return [(r.page_content, r.metadata) for r in results]

    def bm25_search(self, query: str, k: int = 5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_idx[:k]:
            chunk = self.chunks[idx]
            results.append((chunk["text"], {
                "page_numbers": str(chunk["page_numbers"]),
                "chunk_type": chunk["chunk_type"],
            }))
        return results

    def hybrid_search(self, query: str, k: int = 4):
        vec_results = self.vector_search(query, k=5)
        bm25_results = self.bm25_search(query, k=5)

        # Reciprocal rank fusion -- combine by rank position, not raw score,
        # since vector similarity and BM25 scores aren't on comparable scales
        scores = {}
        docs_by_key = {}
        for rank, (text, meta) in enumerate(vec_results):
            key = text[:200]
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs_by_key[key] = (text, meta)
        for rank, (text, meta) in enumerate(bm25_results):
            key = text[:200]
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            docs_by_key[key] = (text, meta)

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [docs_by_key[k] for k in ranked_keys[:k]]