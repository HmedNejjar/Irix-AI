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
        else: return ""
    except Exception as e:
        return f"Summarization failed: {str(e)}"


def router(usr_inpt: str, history: list, routing_prompt: str,summ_prompt: str, model: str, summary_model: str):
    summary, recent = get_router_context(history)

    router_input = {
        "user_input": usr_inpt,
        "conversation_summary": summary,
        "recent_messages": recent
    }

    messages = [
        {"role": "system", "content": routing_prompt},
        {"role": "user", "content": json.dumps(router_input, ensure_ascii=False)}
    ]

    # Get routing decision with fallback
    response = None
    try:
        response = ollama.chat(model=model, messages=messages)
    except Exception as e:
        print(f"⚠️ Router model error: {e}")
        try:
            installed = [m.model for m in ollama.list().models]
            fallback = next((m for m in ["phi3:mini"] if m in installed), installed[0] if installed else None)
            if fallback:
                print(f"ℹ️ Falling back to '{fallback}' for routing.")
                response = ollama.chat(model=fallback, messages=messages)
        except Exception as e2:
            print(f"⚠️ Fallback routing failed: {e2}")
            return {"path": "deliberate", "needs_search": False, "complexity": 3}

    if not response or not response.message.content:
        return {"path": "deliberate", "needs_search": False, "complexity": 3}

    route = extract_json_robust(response.message.content)

    if not route or not isinstance(route, dict):
        return {"path": "deliberate", "needs_search": False, "complexity": 3}

    # Normalize: needs_search is a clean flag, not a path value
    needs_search = route.get("needs_search", False) or route.get("intent") == "search"
    route["needs_search"] = needs_search
    route.pop("intent_search", None)  # clean up any stray keys

    # Search pipeline: fetch → summarize → attach — happens HERE, before agents
    if needs_search:
        print(f"🔍 Web search triggered for: {usr_inpt}")
        raw = fetch_web_results(usr_inpt)
        summarized = summarize_web_results(raw, summ_prompt, usr_inpt, summary_model)
        route["web_context"] = summarized
        route["path"] = "deliberate"  # search always implies deliberate
        print(f"✅ Web context ready ({len(summarized)} chars)")

    print(f"Router: {route}")
    return route