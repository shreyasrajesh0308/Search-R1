import argparse
import json
import re
import os
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError(
        "rank-bm25 is required for bm25_okapi_server. Install with: pip install rank-bm25"
    ) from e


def _load_json_array(corpus_path: str) -> List[Dict]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(corpus_path: str) -> List[Dict]:
    data = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _normalize_docs(raw_docs: List[Dict]) -> List[Dict]:
    normalized = []
    for d in raw_docs:
        if 'contents' in d and isinstance(d['contents'], str):
            contents = d['contents']
        else:
            title = d.get('title', '')
            text = d.get('text', d.get('body', ''))
            # Fallback: if neither exists, stringify the whole dict
            if not title and not text:
                contents = json.dumps(d, ensure_ascii=False)
            else:
                title = 'No title.' if not title else title
                text = 'No snippet available.' if not text else text
                contents = f'"{title}"\n{text}'
        normalized.append({'contents': contents})
    return normalized


def load_corpus_flexible(corpus_path: str) -> List[Dict]:
    # Try JSON array first (e.g., [ {"title":..., "text":...}, ... ])
    try:
        raw = _load_json_array(corpus_path)
        if isinstance(raw, list):
            return _normalize_docs(raw)
    except Exception:
        pass

    # Fallback to JSONL lines
    raw = _load_jsonl(corpus_path)
    return _normalize_docs(raw)


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


class OkapiBM25Indexer:
    def __init__(self, corpus: List[Dict]):
        # Expect each item has key 'contents' (e.g., "\"Title\"\nBody...")
        self.docs = corpus
        self.doc_texts = [doc.get("contents", "") for doc in self.docs]
        self.doc_tokens = [tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(self.doc_tokens)

    def search(self, query: str, topk: int) -> List[Dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # Arg-sort topk
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        results = []
        for i in top_indices:
            results.append(
                {
                    "document": {
                        # Match the format used across this repo: 'contents' string holding title+body
                        "contents": self.doc_texts[i]
                    },
                    "score": float(scores[i]),
                }
            )
        return results


class SearchRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = True  # kept for API compatibility; always returns scores here


def build_app(indexer: OkapiBM25Indexer, default_topk: int) -> FastAPI:
    app = FastAPI(title="Okapi BM25 (rank-bm25) Server")

    @app.post("/retrieve")
    def retrieve(req: SearchRequest):
        k = req.topk if (req.topk and req.topk > 0) else default_topk
        batched = []
        for q in req.queries:
            batched.append(indexer.search(q, k))
        return {"result": batched}

    return app


def main():
    parser = argparse.ArgumentParser(description="Launch a Python-only Okapi BM25 server (no Java)")
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help=(
            "Path to corpus. Supports JSON array with title/text or JSONL with 'contents'."
        ),
    )
    parser.add_argument("--topk", type=int, default=3, help="Results per query")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    corpus = load_corpus_flexible(args.corpus_path)
    indexer = OkapiBM25Indexer(corpus)
    app = build_app(indexer, args.topk)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
