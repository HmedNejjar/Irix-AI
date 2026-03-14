import json
import ollama

   #Function to save history in a json file
def saveHistory(file: str, history: list) -> None:
    with open(file, "w") as f:
        json.dump(history, f, indent=2)
    
    #Function to load existing history
def loadHistory(file: str, sys_prompt: str) -> list:
    try:
        with open(file, "r") as f:
            history:list  = json.load(f)
            if isinstance(history, list):
                return history
    except FileNotFoundError:
        return [{"role" : "system", "content" : sys_prompt}]
    
    #Function to summarize conversation history when it exceeds trigger length
def summarize(history: list, prompt: str, model:str,trigger:int, max_size: int) -> list:
    syst_prompt =  history[0]
    #Define a flag to check for existing summary 
    has_summary = len(history) > 1 and history[1]["role"] == "system" and  history[1]["content"].startswith("Conversation summary")
    
    if has_summary:                                                     #
        existing_summary, raw_messages = history[1], history[2:]        #   Define boundaries between
    else:                                                               #   summary and raw messages
        existing_summary, raw_messages = None, history[1:]              #

    if len(raw_messages) <= trigger:
        return history
    # Split messages into "overflow" (to be summarized) and "recent" (to keep intact)
    overflow = raw_messages[:-max_size+2]
    recent = raw_messages[-max_size+2:]

    summarization_input = []
    
    if existing_summary:
        summarization_input.append(existing_summary) # Include existing summary if it exists (to build upon it)
    
    summarization_input.extend(overflow) # Add overflow messages to be summarized
    
    summary_message = [{"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(summarization_input, ensure_ascii=False, indent=2)
        }]

    message = ollama.chat(model= model, messages=summary_message, keep_alive=1)
    new_summary = {"role" : "system",
                   "content" : f"Conversation summary : {message.message.content}"}
    
    return [syst_prompt, new_summary] + recent # Return reconstructed history: system prompt + new summary + recent messages