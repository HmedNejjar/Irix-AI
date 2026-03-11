# type: ignore
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"   # Fix SSL EOF on corporate/AV networks

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re
import csv
import json
import time
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
import lm_eval.utils as utils
from IrixAI import IrixSystem
import pandas as pd


# -- Utilities ----------------------------------------------------------------

def strip_emoji(text: str) -> str:
    """Remove emoji and non-ASCII symbols to prevent charmap errors."""
    return text.encode('ascii', errors='ignore').decode('ascii')


def to_float(value) -> float | None:
    """Safely convert extracted string number to float."""
    if value is None:
        return None
    try:
        return float(re.sub(r'[^\d.]', '', str(value)))
    except (ValueError, TypeError):
        return None


def floats_equal(a: float | None, b: float | None) -> bool:
    """Compare two floats with tolerance -- handles 42.0 == 42."""
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


# -- Answer extraction --------------------------------------------------------

def extract_final_number(text: str) -> str | None:
    """
    Priority order:
      1. 'Answer: X' or 'Answer X'
      2. \\boxed{X}
      3. Last dollar amount $X
      4. Last number in text
    """
    text_clean = re.sub(r'(\d),(\d)', r'\1\2', text)  # 70,000 -> 70000

    # Priority 1: explicit Answer label
    match = re.search(r'[Aa]nswer[:\s]+\**\$?\s*(\d+(?:\.\d+)?)', text_clean)
    if match:
        return match.group(1)

    # Priority 2: \boxed{X}
    match = re.search(r'\\boxed\{([^}]+)\}', text_clean)
    if match:
        inner = re.sub(r'[^\d.]', '', match.group(1))
        if inner:
            return inner

    # Priority 3: dollar amounts
    dollar_numbers = re.findall(r'\$\s*(\d+(?:\.\d+)?)', text_clean)
    if dollar_numbers:
        return dollar_numbers[-1]

    # Priority 4: last bare number
    all_numbers = re.findall(r'(\d+(?:\.\d+)?)', text_clean)
    return all_numbers[-1] if all_numbers else None


def ensure_gsm8k_format(output: str) -> str:
    """Guarantee output ends with #### [number] for lm-eval extraction."""
    if re.search(r'####\s*\d+', output):
        return output
    final_number = extract_final_number(output)
    if final_number:
        return output.strip() + f"\n#### {final_number}"
    return output


def extract_predicted_answer(processed_output: str) -> str | None:
    """Pull the number after #### from processed output."""
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', processed_output)
    return match.group(1) if match else None


def extract_expected_answer(doc: dict) -> str | None:
    """Pull the gold answer from a GSM8K doc."""
    answer_field = doc.get("answer", "")
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', str(answer_field))
    if match:
        return match.group(1)
    nums = re.findall(r'(\d+(?:\.\d+)?)', re.sub(r'(\d),(\d)', r'\1\2', str(answer_field)))
    return nums[-1] if nums else None


# -- CSV live-write helper -----------------------------------------------------

CSV_FILE    = "irix_eval_per_question.csv"
CSV_HEADERS = [
    "q_index", "status", "expected", "expected_float",
    "predicted", "predicted_float", "correct",
    "latency_ms", "route_path", "complexity",
    "used_search", "model_used", "answer_length",
    "prompt_tail", "raw_tail"
]

def init_csv():
    """Create CSV with headers if it doesn't exist yet."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def append_csv(row: dict):
    """Append a single row immediately after each question."""
    with open(CSV_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
        writer.writerow(row)


# -- Wrapper ------------------------------------------------------------------

@register_model("irix_pipeline")
class IrixEvalWrapper(LM):
    def __init__(self):
        super().__init__()
        self.irix = IrixSystem()
        self.base_sys_prompt = self.irix.history[0]
        self.question_log: list[dict] = []
        self._q_index  = 0
        self._n_passed = 0
        init_csv()

    def generate_until(self, requests):
        res = []
        for request in requests:
            self._q_index += 1
            prompt     = request.args[0]
            doc        = getattr(request, "doc", {})
            expected   = extract_expected_answer(doc)
            expected_f = to_float(expected)

            self.irix.history = [self.base_sys_prompt]

            t0 = time.time()
            try:
                raw_output = self.irix.process(prompt)
            except Exception as e:
                print(f"[Q{self._q_index}] Warning: Error: {e}")
                raw_output = ""
            latency_ms = int((time.time() - t0) * 1000)

            raw_output_safe = strip_emoji(raw_output)

            processed   = ensure_gsm8k_format(raw_output_safe)
            predicted   = extract_predicted_answer(processed)
            predicted_f = to_float(predicted)

            passed = floats_equal(predicted_f, expected_f)
            if passed:
                self._n_passed += 1

            status = "PASS" if passed else "FAIL"

            sep = "=" * 60
            print(f"\n{sep}")
            print(f"[Q{self._q_index}] {status}  |  "
                  f"expected={expected_f}  predicted={predicted_f}  ({latency_ms}ms)")
            print(f"Score so far: {self._n_passed}/{self._q_index} "
                  f"({self._n_passed / self._q_index * 100:.1f}%)")
            print(f"PROMPT (last 200): ...{strip_emoji(prompt[-200:])}")
            print(f"RAW TAIL (last 300): ...{repr(raw_output_safe[-300:])}")
            print(f"PROCESSED TAIL: ...{repr(processed[-120:])}")
            print(f"{sep}\n")

            last_telemetry = self._read_last_telemetry()
            row = {
                "q_index":         self._q_index,
                "status":          status,
                "expected":        expected,
                "expected_float":  expected_f,
                "predicted":       predicted,
                "predicted_float": predicted_f,
                "correct":         passed,
                "latency_ms":      latency_ms,
                "route_path":      last_telemetry.get("router_output", {}).get("path", "?"),
                "complexity":      last_telemetry.get("router_output", {}).get("complexity", "?"),
                "used_search":     last_telemetry.get("used_search", False),
                "model_used":      last_telemetry.get("model_used", "?"),
                "answer_length":   len(raw_output_safe),
                "prompt_tail":     strip_emoji(prompt[-300:]),
                "raw_tail":        strip_emoji(raw_output_safe[-400:]),
            }

            self.question_log.append(row)
            append_csv(row)  # written immediately -- crash-safe

            res.append(processed)
        return res

    def _read_last_telemetry(self) -> dict:
        try:
            with open(self.irix.TELEMETRY_FILE, "r", encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
        except Exception:
            pass
        return {}

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not supported by Irix-AI MAS.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling loglikelihood not supported.")


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting evaluation of Irix-AI Pipeline...")

    tasks        = ["gsm8k"]
    irix_wrapper = IrixEvalWrapper()

    results = simple_evaluate(
        model=irix_wrapper,
        tasks=tasks,
        num_fewshot=8,
        limit=10,
        random_seed=42
    )

    df_questions = pd.DataFrame(irix_wrapper.question_log)

    agg_rows = []
    for task_name, metrics in results["results"].items():
        row = {"Task": task_name}
        row.update(metrics)
        agg_rows.append(row)
    df_agg = pd.DataFrame(agg_rows)

    df_failures = df_questions[df_questions["correct"] == False][[
        "q_index", "expected", "expected_float", "predicted", "predicted_float",
        "route_path", "complexity", "model_used", "latency_ms", "prompt_tail", "raw_tail"
    ]]

    # per_question already written live; overwrite with final clean copy
    df_questions.to_csv("irix_eval_per_question.csv", index=False, encoding='utf-8')
    df_agg.to_csv(      "irix_eval_aggregate.csv",    index=False, encoding='utf-8')
    df_failures.to_csv( "irix_eval_failures.csv",     index=False, encoding='utf-8')

    total   = len(df_questions)
    correct = df_questions["correct"].sum()
    sep     = "=" * 60
    print(f"\n{sep}")
    print(f"Evaluation complete:  {correct}/{total}  ({correct / total * 100:.1f}%)")
    print(f"Results saved to 3 CSV files: per_question, aggregate, failures")
    print(f"{sep}\n")
    print(utils.make_table(results))