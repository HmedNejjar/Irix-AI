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
        self.PROMPT_FILE     = "_prompts.json"
        self.HISTORY_FILE    = "history.json"
        self.TELEMETRY_FILE  = "_router_telemetry.jsonl"

        # Models
        self.AGENT_MODEL     = "qwen3:1.7b"
        self.LIGHT_MODEL     = "qwen2.5:7b"
        self.HEAVY_MODEL     = "qwen3:8b"
        self.SUMMARY_MODEL   = "qwen3:1.7b"
        self.CLASSIFIER_MODEL = "qwen3:1.7b"   # single model for both classifiers

        # Memory config — SUMMARY_TRIGGER must be > HISTORY_MAXSIZE
        self.HISTORY_MAXSIZE = 14
        self.SUMMARY_TRIGGER = 20

        with open(self.PROMPT_FILE, 'r', encoding= 'utf-8') as f:
            self.prompts = json.load(f)

        self.sys_prompt           = self.prompts["system"]["default"]
        self.summary_prompt       = self.prompts["summary"]["conversation"]
        self.web_prompt           = self.prompts["summary"]["web_search"]
        self.complexity_prompt    = self.prompts["router"]["complexity_prompt"]
        self.search_prompt        = self.prompts["router"]["search_prompt"]
        self.eval_prompt          = self.prompts["self_eval"]["prompt"]
        self.agents_prompt        = self.prompts["agents"]
        self.heavy_prompt         = self.prompts["heavy_synthesis"]["prompt"]

        # Agents
        self.all_agents = [
            Agent("edge_case_agent",   self.AGENT_MODEL, self.agents_prompt["edge_case"]),
            Agent("constraints_agent", self.AGENT_MODEL, self.agents_prompt["constraints"]),
            Agent("builder_agent",     self.AGENT_MODEL, self.agents_prompt["builder"])
        ]
        self.builder_agent = self.all_agents[2]

        self.history = loadHistory(self.HISTORY_FILE, self.sys_prompt)

    def process(self, usr_inpt: str) -> str:
        self.history.append({"role": "user", "content": usr_inpt})

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

        route = router(
            usr_inpt,
            self.history,
            self.complexity_prompt,
            self.search_prompt,
            self.web_prompt,
            self.CLASSIFIER_MODEL,
            self.SUMMARY_MODEL
        )

        start_time = time.time()

        if route["path"] == "direct":
            model = self.LIGHT_MODEL
            # Inject web context into history snapshot if search fired on a simple query
            messages = self._build_direct_messages(route)
            print(f"Irix({model}): ", end='', flush=True)
            bot_answer_content = ""
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                if chunk.message.content:
                    bot_answer_content += chunk.message.content
                    print(chunk.message.content, end='', flush=True)
            print()

        else:
            web_context = route.get("web_context")
            complexity  = route.get("complexity", 3)

            conv_summary = None
            if len(self.history) > 1 and self.history[1]["role"] == "system" and \
               self.history[1]["content"].startswith("Conversation summary"):
                conv_summary = self.history[1]["content"]

            agent_context = {
                "conversation_summary": conv_summary,
                "route_decision":       route,
                "web_context":          web_context
            }

            # Full panel for complexity >= 5, Builder-only otherwise
            if complexity >= 5:
                active_agents = self.all_agents
                print(f" Full agent panel (complexity={complexity})")
            else:
                active_agents = [self.builder_agent]
                print(f" Builder-only path (complexity={complexity})")

            agents_outputs = []
            with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
                futures = [executor.submit(a.run, usr_inpt, agent_context) for a in active_agents]
                for future in as_completed(futures):
                    agents_outputs.append(future.result())

            model = self.HEAVY_MODEL
            synth_payload = {
                "question":    usr_inpt,
                "agents":      agents_outputs,
                "web_context": web_context
            }
            synth_messages = [
                {"role": "system", "content": self.heavy_prompt},
                {"role": "user",   "content": json.dumps(synth_payload, ensure_ascii=False)}
            ]

            print(f"Irix({model}): ", end='', flush=True)
            bot_answer_content = ""
            for chunk in ollama.chat(model=model, messages=synth_messages + self.history[-2:], stream=True):
                if chunk.message.content:
                    bot_answer_content += chunk.message.content
                    print(chunk.message.content, end='', flush=True)
            print()

        latency = int((time.time() - start_time) * 1000)
        self._log_and_eval(usr_inpt, route, model, bot_answer_content, agents_outputs, latency)

        self.history.append({"role": "assistant", "content": bot_answer_content})
        return bot_answer_content if bot_answer_content else ""

    def _build_direct_messages(self, route: dict) -> list:
        """Build message list for direct path, injecting web context if present."""
        messages = list(self.history)
        web_context = route.get("web_context")
        if web_context:
            # Inject as system message just before the last user turn
            messages.insert(-1, {
                "role": "system",
                "content": f"Web search context (use this to answer):\n{web_context}"
            })
        return messages

    def _log_and_eval(self, usr_inpt, route, model, answer, agents_out, latency):
        telemetry = {
            "timestamp":     time.time(),
            "user_input":    usr_inpt,
            "router_output": route,
            "model_used":    model,
            "answer_length": len(answer),
            "agents":        agents_out,
            "latency_ms":    latency,
            "used_search":   bool(route.get("web_context"))
        }

        if route["path"] == "deliberate":
            eval_result = self_eval(usr_inpt, self.eval_prompt, self.CLASSIFIER_MODEL, route, answer)
            if eval_result:
                telemetry["self_eval"] = eval_result

        log_telemetry(self.TELEMETRY_FILE, telemetry)