import ollama
import json

class Agent:
    def __init__(self, role: str, model: str, sys_prompt: str) -> None:
        self.role = role
        self.model = model
        self.sys_default = sys_prompt
        
    def run(self, usr_inpt: str, context) -> dict:
        messages = [{"role": "system", "content" : self.sys_default},
                    {"role" : "user", "content" : usr_inpt}]
        
        if context:
            messages.insert(1, {"role" : "system", "content" : "Context: "+ json.dumps(context, ensure_ascii=False, indent=2)})
            
        response = ollama.chat(model=self.model, messages=messages,keep_alive=0)
        content = response.message.content
        
        if content:
            content = content.strip()
            return {"agent" : self.role, "result": content}
        return {"agent" : self.role, "result": "no conclusion, skip"}