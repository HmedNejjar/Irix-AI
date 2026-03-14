import json
import ollama
from __extract_json_robust import extract_json_robust


    #Function to evaluate router decisions
def self_eval(usr_input: str, eval_prompt: str, model:str,  route: dict, answer: str):
    content = eval_prompt.format(
        question=usr_input,
        use_heavy=(route["path"] == "deliberate"),
        answer=answer
    )

    message = [{"role": "system", "content": content}]
    response = ollama.chat(model=model, messages=message, keep_alive=1)

    result = extract_json_robust(response.message.content)
    return result if isinstance(result, dict) else None

    #Function to log bot's output
def log_telemetry(file: str, record: dict) -> None:
    with open(file, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
