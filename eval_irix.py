# type: ignore
import re
import json
import math
import time
import sys
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
    # GSM8K stores answers as "... #### "
    match = re.search(r'####\s*(\d+(?:\.\d+)?)', str(answer_field))
    if match:
        return match.group(1)
    # Fallback: last number in answer field
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', re.sub(r'(\d),(\d)', r'\1\2', str(answer_field)))
    return nums[-1] if nums else None


def _as_float(value: str | None) -> float | None:
    """Convert a numeric string to a float, or return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ── Wrapper ──────────────────────────────────────────────────────────────────

@register_model("irix_pipeline")
class IrixEvalWrapper(LM):
    def __init__(self):
        super().__init__()
        # Force UTF-8 stdout so emoji don't crash on Windows consoles
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        self.irix = IrixSystem()
        self.base_sys_prompt = self.irix.history[0]
        self.question_log: list[dict] = []   # collects per-question records
        self._q_index = 0
        self.filename = "irix_eval_results.xlsx"

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
                print(f"[Q{self._q_index}] [ERROR] {e}")
                raw_output = ""
            latency_ms = int((time.time() - t0) * 1000)

            processed     = ensure_gsm8k_format(raw_output)
            predicted_raw  = extract_predicted_answer(processed)
            expected_raw   = expected
            predicted      = _as_float(predicted_raw)
            expected       = _as_float(expected_raw)
            passed         = (math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6)
                              if (predicted is not None and expected is not None) else False)

            # ── Per-question console output ───────────────────────────────
            status = "✅" if passed else "❌"
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

            self._flush_to_excel()
            res.append(processed)
        return res

    def _flush_to_excel(self) -> None:
        """Rewrite the Excel file with all questions logged so far. Called after every question."""
        try:
            df_questions = pd.DataFrame(self.question_log)
            df_failures  = df_questions[df_questions["correct"] == False][[
                "q_index", "expected", "predicted", "route_path",
                "complexity", "model_used", "latency_ms", "prompt_tail", "raw_tail"
            ]]
            total   = len(df_questions)
            correct = int(df_questions["correct"].sum())
            df_agg  = pd.DataFrame([{
                "questions_so_far": total,
                "correct":          correct,
                "accuracy_pct":     round(correct / total * 100, 1) if total else 0.0,
            }])
            with pd.ExcelWriter(self.filename, engine="openpyxl") as writer:
                df_questions.to_excel(writer, sheet_name="Per-Question",  index=False)
                df_agg.to_excel(      writer, sheet_name="Aggregate",     index=False)
                df_failures.to_excel( writer, sheet_name="Failures Only", index=False)
            print(f"[SAVED] Q{self._q_index} flushed to {self.filename}  ({correct}/{total})")
        except Exception as e:
            print(f"[WARN] Excel flush failed: {e}")

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

    # ── Sheets written live after each question — just print final summary ────
    df_questions = pd.DataFrame(irix_wrapper.question_log)
    total   = len(df_questions)
    correct = int(df_questions["correct"].sum())
    print(f"\n{'='*60}")
    print(f"[DONE] Evaluation complete:  {correct}/{total}  ({correct/total*100:.1f}%)")
    print(f"[OUT]  Results saved to {irix_wrapper.filename}  (3 sheets: Per-Question, Aggregate, Failures Only)")
    print(f"{'='*60}\n")
    print(utils.make_table(results))