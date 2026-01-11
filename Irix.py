"""Irix v1.6.3: Fixed minor bugs & clean structure of code"""

from Agents import Agent
from router import router
from memory import saveHistory, loadHistory, summarize
from telemetry import log_telemetry, self_eval

import ollama
import json
import time

PROMPT_FILE = "_prompts.json"
HISTORY_FILE = "history.json"
TELEMETRY_FILE = "_router_telemetry.jsonl"
AGENT = "qwen3:1.7b"
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
eval_prompt = prompts["self_eval"]["prompt"]
agents_prompt = prompts["agents"]
heavy_prompt = prompts["heavy_synthesis"]["prompt"]

agents = [Agent("edge_case_agent", AGENT, agents_prompt["edge_case"]),
          Agent("constraints_agent", AGENT, agents_prompt["constraints"]),
          Agent("builder_agent", AGENT, agents_prompt["builder"])]

    #Handles the chatbot interaction by selecting the appropriate model, generating a response, and updating conversation history.
def chatbot(usr_inpt: str,lm: str,hm: str,history: list) -> None:
    route = router(usr_inpt, history, routing_prompt, ROUTER_MODEL)
    if route["path"] == "direct":
        start = time.time()
        
        model = lm
        
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
    
        latency = int((time.time() - start)*1000)
    else:
        agents_outputs = []
        
        for agent in agents:
            result = agent.run(usr_inpt, context= None)
            agents_outputs.append(result)
        
        synth_payload = {"question" : usr_inpt, "agents" : agents_outputs}
         
        synth_message = [{"role" : "system", "content" : heavy_prompt},
                         {"role" : "user", "content" : json.dumps(synth_payload, ensure_ascii=False, indent= 2)}]
        
        start = time.time()
        
        model = hm
        
        print(f"Irix({model}): ", end='', flush=True)
        bot_answer_content = ""
        bot_answer = ollama.chat(model= model, messages=synth_message + history[-2:], stream=True)
        
        for chunk in bot_answer:                                
            if chunk.message.content:                           
                bot_answer_content += chunk.message.content     
                print(chunk.message.content, end='', flush=True)
        print()
    
        bot_message = {"role" : "assistant",
                        "content" : bot_answer_content}
        history.append(bot_message)
    
        latency = int((time.time() - start)*1000)
        
    telemetry = {
    "timestamp": time.time(),
    "user_input": usr_inpt,
    "router_output": route,
    "model_used": model,
    "answer_length": len(bot_answer_content),
    "latency_ms": latency
    }

    eval_result = self_eval(usr_inpt,eval_prompt, ROUTER_MODEL, route, bot_answer_content)
    if eval_result:
        telemetry["self_eval"] = eval_result
    log_telemetry(TELEMETRY_FILE, telemetry)

    
    #Main function that runs the Irix AI assistant
def main() -> None:
    system_prompt = sys_prompt

    light_model, heavy_model = LIGHT_MODEL , HEAVY_MODEL
    history = loadHistory(HISTORY_FILE, system_prompt)

    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() in ('exit', 'quit', 'bye'):
            print("Irix: cya👋"); break
            
        user_message = {"role" : "user",
                        "content" : user_prompt}
        history.append(user_message)
        chatbot(user_prompt, light_model, heavy_model, history)
        history = summarize(history, summary_prompt, SUMMARY_MODEL, SUMMARY_TRIGGER, HISTORY_MAXSIZE)
        saveHistory(HISTORY_FILE, history)


if __name__ == "__main__":
    main()
