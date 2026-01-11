import json
import ollama
from __extract_json_robust import extract_json_robust

def get_router_context(history: list) -> tuple:
    summary, recent = None, []
    
    if len(history) > 1 and history[1]["role"] == "system" and history[1]["content"].startswith("Conversation summary"):
        summary = history[1]["content"]
        
        for msg in reversed(history):
            if msg["role"] == "user": recent.append(msg)
            if len(recent) == 4: break
        recent.reverse()
    return summary, recent


def router(usr_inpt: str, history: list, routing_prompt: str, model: str):
    summary, recent = get_router_context(history)
    
    router_input = {"user_input" : usr_inpt,
                    "conversation_summary" : summary,
                    "recent_messages" : recent}
    
    messages = [
        {"role": "system",
         "content": routing_prompt},
        
        {"role": "user",
         "content": json.dumps(router_input, ensure_ascii=False)}
    ]
    
    response = ollama.chat(model, messages=messages)
    
    if not response.message.content:
        return {"path": "deliberate"}
    
    # Try to extract clean JSON
    clean_json = extract_json_robust((response.message.content))
    
    if clean_json and isinstance(clean_json, dict):
        print(clean_json) #debugging
        return clean_json
    else:
        return {"path": "deliberate"}