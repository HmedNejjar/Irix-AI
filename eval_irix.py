# type: ignore
import re
import sys
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
import lm_eval.utils as utils
from IrixAI import IrixSystem
import pandas as pd


def extract_final_number(text: str) -> str | None:
    """
    Extract the final numeric answer from model output.
    Strategy: look for explicit answer signal phrases LAST in the text,
    since models state intermediate numbers before the final answer.
    """
    # Clean commas from numbers like 70,000 → 70000 for matching
    text_clean = re.sub(r'(\d),(\d)', r'\1\2', text)

    # Priority 1: explicit answer declaration — last occurrence wins
    # e.g. "Answer: $70,000", "**Answer:** $70,000", "profit is $70,000"
    answer_signals = [
        r'(?:\*{0,2}answer\*{0,2}[\s:—\-]+)\$?\s*(\d+(?:\.\d+)?)',
        r'(?:profit|total|result|therefore|thus|so)\s+(?:is|=|:)\s*\$?\s*(\d+(?:\.\d+)?)',
        r'=\s*\*{0,2}\$?\s*(\d+(?:\.\d+)?)\*{0,2}\.?\s*$',
    ]
    for pattern in answer_signals:
        matches = list(re.finditer(pattern, text_clean, re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1)  # Last match = final answer

    # Priority 2: last number preceded by $ sign
    dollar_numbers = re.findall(r'\$\s*(\d+(?:\.\d+)?)', text_clean)
    if dollar_numbers:
        return dollar_numbers[-1]

    # Fallback: last standalone integer in text
    all_numbers = re.findall(r'\b(\d+)\b', text_clean)
    return all_numbers[-1] if all_numbers else None


def ensure_gsm8k_format(output: str) -> str:
    """
    Ensures output ends with #### [number].
    If already present, return as-is.
    If missing, extract final answer and append it.
    """
    if re.search(r'####\s*\d+', output):
        return output  # Already correctly formatted

    final_number = extract_final_number(output)
    if final_number:
        return output.strip() + f"\n#### {final_number}"

    return output  # Can't extract — will score 0


@register_model("irix_pipeline")
class IrixEvalWrapper(LM):
    def __init__(self):
        super().__init__()
        self.irix = IrixSystem()
        self.base_sys_prompt = self.irix.history[0]

    def generate_until(self, requests):
        res = []
        for request in requests:
            prompt = request.args[0]

            # Isolate each benchmark question — no history bleed
            self.irix.history = [self.base_sys_prompt]

            try:
                output = self.irix.process(prompt)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                output = ""

            # Safety net: guarantee #### format regardless of prompt compliance
            processed = ensure_gsm8k_format(output)

            # Debug
            print("\n--- RAW OUTPUT ---")
            print(repr(output[-300:]))  # Last 300 chars only
            print("--- PROCESSED TAIL ---")
            print(repr(processed[-100:]))
            print("----------------------\n")

            res.append(processed)
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not supported by Irix-AI MAS.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling loglikelihood not supported.")


# --- Run evaluation ---
if __name__ == "__main__":
    print("Starting evaluation of Irix-AI Pipeline...")

    tasks = ["gsm8k"]
    irix_wrapper = IrixEvalWrapper()

    results = simple_evaluate(
        model=irix_wrapper,
        tasks=tasks,
        num_fewshot=8,
        limit=None,
        random_seed=42
    )

    # --- Export to Excel ---
    export_data = []
    for task_name, metrics in results["results"].items():
        row = {"Task": task_name}
        row.update(metrics)
        export_data.append(row)

    df = pd.DataFrame(export_data)
    filename = "irix_eval_results.xlsx"
    df.to_excel(filename, index=False)

    print(f"\n✅ Evaluation complete! Results saved to {filename}")
    print(utils.make_table(results))