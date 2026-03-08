import json
import ollama
from __extract_json_robust import extract_json_robust


def get_router_context(history: list) -> tuple:
    summary, recent = None, []

    if len(history) > 1 and history[1]["role"] == "system" and history[1]["content"].startswith("Conversation summary"):
        summary = history[1]["content"]
        start = 2
    else:
        start = 1

    exchanges = []
    msgs = history[start:]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] in ("user", "assistant"):
            exchanges.append(msgs[i])
        if len(exchanges) == 8:
            break
    recent = list(reversed(exchanges))

    return summary, recent


def complexity_classifier(usr_inpt: str, history: list, prompt: str, model: str) -> int:
    """Stage 1: Classify complexity as score 1–5. Returns int."""
    summary, recent = get_router_context(history)

    payload = {
        "user_input": usr_inpt,
        "conversation_summary": summary,
        "recent_messages": recent
    }

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]

    try:
        response = ollama.chat(model=model, messages=messages)
        result = extract_json_robust(response.message.content)
        if isinstance(result, dict):
            score = result.get("complexity", 3)
            if isinstance(score, int) and 1 <= score <= 5:
                return score
    except Exception as e:
        print(f"⚠️ Complexity classifier error: {e}")

    return 3  # safe default


def search_classifier(usr_inpt: str, prompt: str, model: str) -> bool:
    """Stage 2: Decide if web search is needed. Returns bool."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"user_input": usr_inpt}, ensure_ascii=False)}
    ]

    try:
        response = ollama.chat(model=model, messages=messages)
        result = extract_json_robust(response.message.content)
        if isinstance(result, dict):
            return bool(result.get("needs_search", False))
    except Exception as e:
        print(f"⚠️ Search classifier error: {e}")

    return False  # safe default: don't search


def fetch_web_results(query: str) -> str:
    """Fetch raw search results. Returns condensed string or error message."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Web search unavailable: install 'duckduckgo-search' package."

    search_data = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            for r in results:
                search_data.append(f"Source: {r['href']}\nTitle: {r['title']}\nContent: {r['body']}")
        return "\n\n".join(search_data) if search_data else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


def summarize_web_results(raw_results: str, prompt: str, query: str, model: str) -> str:
    """Summarize raw search results into a clean, concise context block."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Query: {query}\n\nRaw results:\n{raw_results}"}
    ]
    try:
        response = ollama.chat(model=model, messages=messages)
        if response.message.content:
            return response.message.content.strip()
        return ""
    except Exception as e:
        return f"Summarization failed: {str(e)}"


def router(usr_inpt: str, history: list, complexity_prompt: str, search_prompt: str,
           summ_prompt: str, model: str, summary_model: str) -> dict:
    """
    Two-stage router:
      Stage 1 — Complexity Classifier  → score 1–5
      Stage 2 — Search Classifier      → yes/no  (runs for ALL paths)

    Returns route dict with keys: path, complexity, needs_search, web_context
    """

    # ── Stage 1: Complexity ──────────────────────────────────────────────────
    complexity = complexity_classifier(usr_inpt, history, complexity_prompt, model)
    print(f"[Stage 1] Complexity score: {complexity}")

    # ── Stage 2: Search ──────────────────────────────────────────────────────
    needs_search = search_classifier(usr_inpt, search_prompt, model)
    print(f"[Stage 2] Needs search: {needs_search}")

    # ── Path decision ────────────────────────────────────────────────────────
    path = "deliberate" if complexity > 3 else "direct"

    route = {
        "path": path,
        "complexity": complexity,
        "needs_search": needs_search,
        "web_context": None
    }

    # ── Search pipeline (universal — runs regardless of path) ────────────────
    if needs_search:
        print(f"🔍 Web search triggered for: {usr_inpt}")
        raw = fetch_web_results(usr_inpt)
        summarized = summarize_web_results(raw, summ_prompt, usr_inpt, summary_model)
        route["web_context"] = summarized
        print(f"✅ Web context ready ({len(summarized)} chars)")

    print(f"Router final: {route}")
    return route