"""Irix v1.4: Modified the model selection from using keywords to using an LLM based router - semantic routing -
                 which checks the context and selects the the adequate model for reasoning"""


from __extract_json_robust import extract_json_robust
import ollama
import json

PROMPT_FILE = "_prompts.json"
HISTORY_FILE = "history.json"
LIGHT_MODEL = "qwen2.5:7b"
HEAVY_MODEL = "qwen3:8b"
SUMMARY_MODEL = "phi3:mini"
ROUTER_MODEL = "phi3:mini"
HISTORY_MAXSIZE = 14
SUMMARY_TRIGGER = 16

with open (PROMPT_FILE, 'r') as f:
    prompts = json.load(f)

sys_prompt = prompts["system"]["default"]
summary_prompt = prompts["summary"]["conversation"]
routing_prompt = prompts["router"]["prompt"]

    #Function to save history in a json file
def saveHistory(history: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    #Function to load existing history
def loadHistory(sys_prompt: str) -> list:
    try:
        with open(HISTORY_FILE, "r") as f:
            history:list  = json.load(f)
            if isinstance(history, list):
                return history
    except FileNotFoundError:
        return [{"role" : "system", "content" : sys_prompt}]

    #Function to summarize conversation history when it exceeds trigger length
def summarize(history: list) -> list:
    syst_prompt =  history[0]
    #Define a flag to check for existing summary 
    has_summary = len(history) > 1 and history[1]["role"] == "system" and  history[1]["content"].startswith("Conversation summary")
    
    if has_summary:                                                     #
        existing_summary, raw_messages = history[1], history[2:]        #   Define boundaries between
    else:                                                               #   summary and raw messages
        existing_summary, raw_messages = None, history[1:]              #

    if len(raw_messages) <= SUMMARY_TRIGGER:
        return history
    # Split messages into "overflow" (to be summarized) and "recent" (to keep intact)
    overflow = raw_messages[:-HISTORY_MAXSIZE+2]
    recent = raw_messages[-HISTORY_MAXSIZE+2:]

    summarization_input = []
    
    if existing_summary:
        summarization_input.append(existing_summary) # Include existing summary if it exists (to build upon it)
    
    summarization_input.extend(overflow) # Add overflow messages to be summarized
    
    summary_message = [{"role": "system", "content": summary_prompt},
        {
            "role": "user",
            "content": json.dumps(summarization_input, ensure_ascii=False, indent=2)
        }]

    message = ollama.chat(model= SUMMARY_MODEL, messages=summary_message)
    new_summary = {"role" : "system",
                   "content" : f"Conversation summary : {message.message.content}"}
    
    return [syst_prompt, new_summary] + recent # Return reconstructed history: system prompt + new summary + recent messages

    #Function to get context for router to choose which model to use during  conversation
def get_router_context(history: list) -> tuple:
    summary, recent = None, []
    
    if len(history) > 1 and history[1]["role"] == "system" and history[1]["content"].startswith("Conversation summary"):
        summary = history[1]["content"]
        
        for msg in reversed(history):
            if msg["role"] == "user": recent.append(msg)
            if len(recent) == 4: break
        recent.reverse()
    return summary, recent

#Function to let router decide which model to use according to context and history
def router(usr_inpt, history):
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
    
    response = ollama.chat(ROUTER_MODEL, messages=messages)
    
    if not response.message.content:
        return {"use_heavy": False}
    
    # Try to extract clean JSON
    clean_json = extract_json_robust((response.message.content))
    
    if clean_json and isinstance(clean_json, dict):
        print(clean_json) #debugging
        return clean_json
    else:
        return {"use_heavy": True}

    #Handles the chatbot interaction by selecting the appropriate model, generating a response, and updating conversation history.
def chatbot(usr_inpt: str,lm: str,hm: str,history: list) -> None:
    route = router(usr_inpt, history)
    model = hm if route["use_heavy"] else lm
        
    print(f"Irix({model}): ", end='', flush=True)
    bot_answer_content = ""
    bot_answer = ollama.chat(model= model, messages=history, stream=True)
    
    for chunk in bot_answer:                                               #
        if chunk.message.content:                                          # Process streamed response chunks 
            bot_answer_content += chunk.message.content                    # and print each chunk for smooth answer
            print(chunk.message.content, end='', flush=True)               #
    print()

    bot_message = {"role" : "assistant",
                    "content" : bot_answer_content}
    history.append(bot_message)

    #Main function that runs the Irix AI assistant
def main() -> None:
    system_prompt = sys_prompt

    light_model, heavy_model = LIGHT_MODEL , HEAVY_MODEL
    history = loadHistory(system_prompt)

    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() in ('exit', 'quit', 'bye'):
            print("Irix: cya👋"); break
            
        user_message = {"role" : "user",
                        "content" : user_prompt}
        history.append(user_message)
        chatbot(user_prompt, light_model, heavy_model, history)
        history = summarize(history)
        saveHistory(history)


if __name__ == "__main__":
    main()
