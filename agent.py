"""W16 single-agent loop: self-checking RAG research assistant (stdlib only).

A fixed pipeline is insufficient because the agent must decide from intermediate
search results whether the evidence is sufficient or whether another search is necessary.
"""
import json, os, re
from rag_tool import RagTool, default_corpus
import rag_tool as _rt

# Load .env if present (stdlib, no dependency)
if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
    with open(os.path.join(os.path.dirname(__file__), ".env")) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#") and k not in os.environ:
                os.environ[k] = v

MAX_ITERATIONS = 5  # hard stop: the agent never runs indefinitely
TOP_K_CAP = 5  # retrieval result capping

def approx_tokens(s: str) -> int:
    # token accounting: exact API usage unavailable offline, so use closest
    # reliable measure ~ 1 token per 4 chars (stated in README). If OPENAI_API_KEY
    # is set and `openai` lib present, real usage is added on top (see below).
    return max(1, len(s) // 4)

def _groq_json(prompt: str):
    """Call Groq chat API (OpenAI-compatible) with stdlib only. Returns (data, usage_tokens)."""
    import urllib.request
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    body = json.dumps({
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": [{"role": "user", "content": prompt + "\nReturn JSON only."}],
        "max_tokens": 150, "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                 data=body, headers={"Authorization": f"Bearer {key}",
                                                     "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    msg = data["choices"][0]["message"]["content"]
    u = data.get("usage") or {}
    toks = (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) or None
    return json.loads(msg), toks

def llm_decide(question, history, evidence, iteration):
    """Model decides next action. Uses OpenAI if key present, else local policy
    that still branches on intermediate tool results (not a fixed sequence)."""
    ev_txt = "\n".join(f"- [{e['doc_id']}] {e['text']}" for e in evidence) or "(no evidence yet)"
    hist = "\n".join(f"iter{i+1}: q={h['query']} n={h['n_results']} err={h.get('error')}" for i, h in enumerate(history)) or "(none)"
    prompt = (f"Question: {question}\nHistory:\n{hist}\nEvidence:\n{ev_txt}\n"
              f"Respond ONLY with JSON {{\"action\": \"search|answer|clarify\", \"query\": \"...\", \"reason\": \"...\"}}")
    # --- where token usage is recorded (prompt part) ---
    toks = approx_tokens(prompt)
    # 1) Groq (preferred live LLM, stdlib call, real usage tokens)
    try:
        out = _groq_json(prompt)
        if out:
            d, real = out
            return {"action": d.get("action", "answer"), "query": d.get("query", ""),
                    "reason": d.get("reason", ""), "tokens": real or toks + approx_tokens(json.dumps(d))}
    except Exception:
        pass  # fall through
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI
            c = OpenAI()
            r = c.chat.completions.create(model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt + "\nReturn JSON only."}],
                max_tokens=150, response_format={"type": "json_object"})
            toks = (r.usage.prompt_tokens + r.usage.completion_tokens) if r.usage else toks + approx_tokens(r.choices[0].message.content)
            d = json.loads(r.choices[0].message.content)
            return {"action": d.get("action", "answer"), "query": d.get("query", ""),
                    "reason": d.get("reason", ""), "tokens": toks}
        except Exception:
            pass  # fall through to local policy
    # Local fallback policy (model-free but result-dependent, not fixed order):
    toks += 60
    if any(h.get("error") for h in history):
        # search tool failed -> must NOT fabricate; clarify/decline
        return {"action": "clarify", "query": "",
                "reason": "Search tool failed; no evidence, cannot answer confidently.", "tokens": toks}
    best = evidence[0]["score"] if evidence else 0
    if not evidence or best < 2.0:
        if iteration >= 2 or len(question.strip().split()) <= 2:
            return {"action": "clarify", "query": "",
                    "reason": "Evidence too weak after search; need clarification.", "tokens": toks}
        # reformulate: drop stopwords / keep keywords
        kw = [w for w in re.findall(r"[a-z]+", question.lower())
              if w not in {"what","is","the","a","an","of","how","do","does","tell","me","about","please"}]
        return {"action": "search", "query": " ".join(kw) or question,
                "reason": f"Weak evidence (best={best}); retry with keywords.", "tokens": toks}
    return {"action": "answer", "query": "",
            "reason": f"Sufficient evidence (best={best}).", "tokens": toks}

def run_agent(question, tool=None, verbose=False):
    # --- where the agentic loop begins ---
    tool = tool or RagTool(default_corpus())
    evidence, history, total_tokens = [], [], approx_tokens(question)
    tool_calls, iters = 0, 0
    # --- where the stopping condition is enforced (MAX_ITERATIONS) ---
    for i in range(MAX_ITERATIONS):
        iters = i + 1
        # --- where the model decides the next action ---
        d = llm_decide(question, history, evidence, i)
        total_tokens += d.pop("tokens", 0)
        if verbose: print(f"[iter {iters}] {d['action']}: {d['reason']}")
        if d["action"] == "search":
            tool_calls += 1
            try:
                res = tool.search(d["query"])
            except Exception as e:
                # --- where failure injection is handled (agent side) ---
                history.append({"query": d["query"], "n_results": 0, "error": str(e)})
                evidence = []
                total_tokens += 20
                continue
            if not res.get("ok") or not res.get("results"):
                # empty result: NOT a tool failure (error stays None) so the
                # agent may retry with a reformulated query next iteration
                history.append({"query": d["query"], "n_results": 0, "error": None})
                evidence = []
                continue
            # --- where retrieval results are capped (top 5) ---
            capped = sorted(res["results"], key=lambda r: r["score"], reverse=True)[:TOP_K_CAP]
            history.append({"query": d["query"], "n_results": len(capped), "error": None})
            # merge, dedupe by doc_id, keep top 5 overall
            seen = {e["doc_id"]: e for e in evidence}
            for r in capped: seen[r["doc_id"]] = r
            evidence = sorted(seen.values(), key=lambda r: r["score"], reverse=True)[:TOP_K_CAP]
            total_tokens += approx_tokens(json.dumps(capped))
        elif d["action"] == "answer":
            if not evidence or any(h.get("error") for h in history):
                # refuse to fabricate without evidence
                ans = "I couldn't retrieve reliable evidence, so I can't answer confidently. Please try again later."
            else:
                ans = f"Based on {evidence[0]['doc_id']}: {evidence[0]['text']}"
                total_tokens += approx_tokens(ans)
            return {"answer": ans, "iterations": iters, "tool_calls": tool_calls,
                    "tokens": total_tokens, "history": history, "clarified": False}
        else:  # clarify
            q = "Could you clarify your question? " + (d.get("reason") or "")
            total_tokens += approx_tokens(q)
            return {"answer": q, "iterations": iters, "tool_calls": tool_calls,
                    "tokens": total_tokens, "history": history, "clarified": True}
    return {"answer": "Max iterations reached without enough evidence. Could you rephrase?",
            "iterations": iters, "tool_calls": tool_calls, "tokens": total_tokens,
            "history": history, "clarified": True}

if __name__ == "__main__":
    import sys
    if "--fail" in sys.argv: _rt.FAILURE_INJECTION = True
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "What is the refund policy?"
    print(json.dumps(run_agent(q, verbose=True), indent=2))
