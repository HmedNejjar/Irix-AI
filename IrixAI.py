import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama

from Agents import Agent
from router import router
from memory import saveHistory, loadHistory, summarize
from telemetry import log_telemetry, self_eval

class IrixSystem:
    def __init__(self):
        self.PROMPT_FILE = "_prompts.json"
        self.HISTORY_FILE = "history.json"
        self.TELEMETRY_FILE = "_router_telemetry.jsonl"
        self.AGENT_MODEL = "qwen3:1.7b"
        self.LIGHT_MODEL = "qwen2.5:7b"
        self.HEAVY_MODEL = "qwen3:8b"
        self.SUMMARY_MODEL = "qwen3:1.7b"
        self.ROUTER_MODEL = "granite3.1-moe:3b"
        self.HISTORY_MAXSIZE = 14
        self.SUMMARY_TRIGGER = 20 

        # Load prompts
        with open(self.PROMPT_FILE, 'r') as f:
            self.prompts = json.load(f)

        self.sys_prompt = self.prompts["system"]["default"]
        self.summary_prompt = self.prompts["summary"]["conversation"]
        self.routing_prompt = self.prompts["router"]["prompt"]
        self.eval_prompt = self.prompts["self_eval"]["prompt"]
        self.agents_prompt = self.prompts["agents"]
        self.heavy_prompt = self.prompts["heavy_synthesis"]["prompt"]

        # Initialize specialized agents
        self.agents = [
            Agent("edge_case_agent", self.AGENT_MODEL, self.agents_prompt["edge_case"]),
            Agent("constraints_agent", self.AGENT_MODEL, self.agents_prompt["constraints"]),
            Agent("builder_agent", self.AGENT_MODEL, self.agents_prompt["builder"])
        ]
        
        # Load conversation state
        self.history = loadHistory(self.HISTORY_FILE, self.sys_prompt)

    def process(self, usr_inpt: str) -> str:
        user_message = {"role": "user", "content": usr_inpt}
        self.history.append(user_message)
        
        bot_answer_content = self._run_logic(usr_inpt)
        
        self.history = summarize(
            self.history,
            self.summary_prompt,
            self.SUMMARY_MODEL,
            self.SUMMARY_TRIGGER,
            self.HISTORY_MAXSIZE
        )
        saveHistory(self.HISTORY_FILE, self.history)
        
        return bot_answer_content if bot_answer_content else ""

    def _run_logic(self, usr_inpt: str) -> str:
        agents_outputs = None
        route = router(usr_inpt, self.history, self.routing_prompt, self.ROUTER_MODEL)
        
        start_time = time.time()
        
        if route["path"] == "direct":
            model = self.LIGHT_MODEL
            print(f"Irix({model}): ", end='', flush=True)
            bot_answer_content = ""
            response = ollama.chat(model=model, messages=self.history, stream=True)
            for chunk in response:
                if chunk.message.content:
                    bot_answer_content += chunk.message.content
                    print(chunk.message.content, end='', flush=True)
            print()

        else:
            # Pass only the summary string to agents, not full history
            summary_context = None
            if len(self.history) > 1 and self.history[1]["role"] == "system" and self.history[1]["content"].startswith("Conversation summary"):
                summary_context = self.history[1]["content"]

            agent_context = {
                "conversation_summary": summary_context,
                "route_decision": route
            }

            agents_outputs = []
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                futures = [executor.submit(agent.run, usr_inpt, agent_context) for agent in self.agents]
                for future in as_completed(futures):
                    agents_outputs.append(future.result())
            
            model = self.HEAVY_MODEL
            synth_payload = {"question": usr_inpt, "agents": agents_outputs}
            synth_messages = [
                {"role": "system", "content": self.heavy_prompt},
                {"role": "user", "content": json.dumps(synth_payload, ensure_ascii=False)}
            ]
            print(f"Irix({model}): ", end='', flush=True)
            bot_answer_content = ""
            response = ollama.chat(model=model, messages=synth_messages + self.history[-2:], stream=True)
            for chunk in response:
                if chunk.message.content:
                    bot_answer_content += chunk.message.content
                    print(chunk.message.content, end='', flush=True)
            print()

        latency = int((time.time() - start_time) * 1000)
        self._log_and_eval(usr_inpt, route, model, bot_answer_content, agents_outputs, latency)
        
        self.history.append({"role": "assistant", "content": bot_answer_content})
        return bot_answer_content if bot_answer_content else ""

    def _log_and_eval(self, usr_inpt, route, model, answer, agents_out, latency):
        telemetry = {
            "timestamp": time.time(),
            "user_input": usr_inpt,
            "router_output": route,
            "model_used": model,
            "answer_length": len(answer),
            "agents": agents_out,
            "latency_ms": latency
        }

        # only run self-eval on deliberate path — it's meaningless otherwise
        if route["path"] == "deliberate":
            eval_result = self_eval(usr_inpt, self.eval_prompt, self.ROUTER_MODEL, route, answer)
            if eval_result:
                telemetry["self_eval"] = eval_result

        log_telemetry(self.TELEMETRY_FILE, telemetry)