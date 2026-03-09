# type: ignore
import re
import json
import time
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
import lm_eval.utils as utils
from IrixAI import IrixSystem
import pandas as pd


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_final_number(text: str) -> str | None:
    """
    Extract the final numeric answer from model output.
    Tries explicit answer signals first, then dollar amounts, then last number.
    """
    text_clean = re.sub(r'(\d),(\d)', r'\1\2', text)  # 70,000 → 70000

    answer_signals = [
        r'(?:\*{0,2}answer\*{0,2}[\s:—\-]+)\$?\s*(\d+(?:\.\d+)?)',
        r'(?:profit|total|result|therefore|thus|so)\s+(?:is|=|:)\s*\$?\s*(\d+(?:\.\d+)?)',
        r'=\s*\*{0,2}\$?\s*(\d+(?:\.\d+)?)\*{0,2}\.?\s*$',
    ]
    for pattern in answer_signals:
        matches = list(re.finditer(pattern, text_clean, re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1)

    dollar_numbers = re.findall(r'\$\s*(\d+(?:\.\d+)?)', text_clean)
    if dollar_numbers:
        return dollar_numbers[-1]

    all_numbers = re.findall(r'\b(\d+)\b', text_clean)
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
    """Pull the gold answer from a GSM8K doc. Handles both answer formats."""
    answer_field = doc.get("answer", "")
    # GSM8K stores answers as "... #### 42"
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', str(answer_field))
    if match:
        return match.group(1)
    # Fallback: last number in answer field
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', re.sub(r'(\d),(\d)', r'\1\2', str(answer_field)))
    return nums[-1] if nums else None


# ── Wrapper ──────────────────────────────────────────────────────────────────

@register_model("irix_pipeline")
class IrixEvalWrapper(LM):
    def __init__(self):
        super().__init__()
        self.irix = IrixSystem()
        self.base_sys_prompt = self.irix.history[0]
        self.question_log: list[dict] = []   # collects per-question records
        self._q_index = 0

    def generate_until(self, requests):
        res = []
        for request in requests:
            self._q_index += 1
            prompt    = request.args[0]
            doc       = getattr(request, "doc", {})          # lm-eval attaches the raw doc
            expected  = extract_expected_answer(doc)

            # Isolate each question — no history bleed between benchmark items
            self.irix.history = [self.base_sys_prompt]

            t0 = time.time()
            try:
                raw_output = self.irix.process(prompt)
            except Exception as e:
                print(f"[Q{self._q_index}] ⚠️  Error: {e}")
                raw_output = ""
            latency_ms = int((time.time() - t0) * 1000)

            processed  = ensure_gsm8k_format(raw_output)
            predicted  = extract_predicted_answer(processed)
            passed     = (predicted == expected) if (predicted and expected) else False

            # ── Per-question console output ───────────────────────────────
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{'='*60}")
            print(f"[Q{self._q_index}] {status}  |  expected={expected}  predicted={predicted}  ({latency_ms}ms)")
            print(f"PROMPT (last 200): ...{prompt[-200:]}")
            print(f"RAW TAIL (last 300): ...{repr(raw_output[-300:])}")
            print(f"PROCESSED TAIL: ...{repr(processed[-120:])}")
            print(f"{'='*60}\n")

            # ── Record for Excel export ───────────────────────────────────
            # Grab route info if IrixAI stored it (best-effort)
            last_telemetry = self._read_last_telemetry()
            self.question_log.append({
                "q_index":        self._q_index,
                "status":         "PASS" if passed else "FAIL",
                "expected":       expected,
                "predicted":      predicted,
                "correct":        passed,
                "latency_ms":     latency_ms,
                "route_path":     last_telemetry.get("router_output", {}).get("path", "?"),
                "complexity":     last_telemetry.get("router_output", {}).get("complexity", "?"),
                "used_search":    last_telemetry.get("used_search", False),
                "model_used":     last_telemetry.get("model_used", "?"),
                "answer_length":  len(raw_output),
                "raw_tail":       raw_output[-400:],
                "prompt_tail":    prompt[-300:],
            })

            res.append(processed)
        return res

    def _read_last_telemetry(self) -> dict:
        """Read the last line from the telemetry JSONL — best-effort."""
        try:
            with open(self.irix.TELEMETRY_FILE, "r") as f:
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting evaluation of Irix-AI Pipeline...")

    tasks       = ["gsm8k"]
    irix_wrapper = IrixEvalWrapper()

    results = simple_evaluate(
        model=irix_wrapper,
        tasks=tasks,
        num_fewshot=8,
        limit=10,
        random_seed=42
    )

    # ── Sheet 1: per-question breakdown ──────────────────────────────────────
    df_questions = pd.DataFrame(irix_wrapper.question_log)

    # ── Sheet 2: aggregate metrics ───────────────────────────────────────────
    agg_rows = []
    for task_name, metrics in results["results"].items():
        row = {"Task": task_name}
        row.update(metrics)
        agg_rows.append(row)
    df_agg = pd.DataFrame(agg_rows)

    # ── Sheet 3: failure summary — only failed questions ─────────────────────
    df_failures = df_questions[df_questions["correct"] == False][[
        "q_index", "expected", "predicted", "route_path",
        "complexity", "model_used", "latency_ms", "prompt_tail", "raw_tail"
    ]]

    filename = "irix_eval_results.xlsx"
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_questions.to_excel(writer, sheet_name="Per-Question",  index=False)
        df_agg.to_excel(      writer, sheet_name="Aggregate",     index=False)
        df_failures.to_excel( writer, sheet_name="Failures Only", index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    total   = len(df_questions)
    correct = df_questions["correct"].sum()
    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete:  {correct}/{total}  ({correct/total*100:.1f}%)")
    print(f"📄 Results saved to {filename}  (3 sheets: Per-Question, Aggregate, Failures Only)")
    print(f"{'='*60}\n")
    print(utils.make_table(results))