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

    # Collect last 4 full exchanges (user + assistant pairs), not just user turns
    exchanges = []
    msgs = history[start:]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] in ("user", "assistant"):
            exchanges.append(msgs[i])
        if len(exchanges) == 8:  # 4 pairs
            break
    recent = list(reversed(exchanges))

    return summary, recent


def router(usr_inpt: str, history: list, routing_prompt: str, model: str):
    summary, recent = get_router_context(history)
    
    router_input = {"user_input": usr_inpt,
                    "conversation_summary": summary,
                    "recent_messages": recent}
    
    messages = [
        {"role": "system", "content": routing_prompt},
        {"role": "user", "content": json.dumps(router_input, ensure_ascii=False)}
    ]
    
    response = ollama.chat(model, messages=messages)
    
    if not response.message.content:
        return {"path": "deliberate"}
    
    clean_json = extract_json_robust(response.message.content)
    
    if clean_json and isinstance(clean_json, dict):
        print(clean_json)  # debugging
        return clean_json
    else:
        return {"path": "deliberate"}