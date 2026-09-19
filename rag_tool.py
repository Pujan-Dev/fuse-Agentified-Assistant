"""W15 baseline: simple local RAG/search tool (TF-IDF, stdlib only). Reused as-is by W16 agent."""
import math, re
from collections import Counter

FAILURE_INJECTION = False  # toggled by agent.py / eval; True forces tool failure

def _tokens(t): return re.findall(r"[a-z0-9]+", t.lower())

class RagTool:
    def __init__(self, docs: dict):
        self.docs = docs  # {doc_id: text}
        # precompute idf
        N = len(docs)
        df = Counter()
        self._tok = {k: _tokens(v) for k, v in docs.items()}
        for toks in self._tok.values():
            for w in set(toks): df[w] += 1
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def _score(self, q_toks, d_toks):
        tf = Counter(d_toks)
        return sum(tf[w] * self.idf.get(w, 0) for w in q_toks)

    def search(self, query: str, top_k: int = 10):
        # --- where failure injection is handled (tool side) ---
        if FAILURE_INJECTION:
            raise TimeoutError("Simulated search timeout (FAILURE_INJECTION=True)")
        if not query or not query.strip():
            return {"ok": False, "error": "empty query", "results": []}
        q = _tokens(query)
        scored = sorted(
            ((self._score(q, dt), did) for did, dt in self._tok.items()),
            reverse=True)
        out = []
        for s, did in scored[:top_k]:
            if s <= 0: continue
            out.append({"doc_id": did, "score": round(s, 3),
                        "text": self.docs[did][:500]})
        return {"ok": True, "results": out}


def default_corpus():
    return {
        "doc1": "The refund policy allows returns within 30 days of purchase with a receipt. Refunds are issued to the original payment method.",
        "doc2": "Shipping takes 3-5 business days for standard delivery. Express shipping delivers in 1-2 business days. Shipping is free over $50.",
        "doc3": "To reset your password, go to Settings > Account > Reset password. A reset link is emailed and expires in 24 hours.",
        "doc4": "The library opening hours are 9am to 8pm Monday to Friday, and 10am to 4pm on Saturday. Closed on Sunday.",
        "doc5": "Machine learning models include regression, classification, and clustering. RAG combines retrieval with generation for grounded answers.",
        "doc6": "Warranty covers manufacturing defects for 12 months. It does not cover accidental damage or water damage.",
        "doc7": "Contact support at support@example.com or call 555-0100 between 9am and 5pm on weekdays.",
        "doc8": "Python virtual environments isolate dependencies. Create one with python -m venv env and activate it before installing packages.",
    }
