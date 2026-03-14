# type: ignore
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def strip_emoji(text: str) -> str:
    return text.encode('ascii', errors='ignore').decode('ascii')


LETTER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_MAP    = {v: k for k, v in LETTER_MAP.items()}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_choice_letter(text: str) -> str | None:
    """
    Priority:
      1. Explicit 'Answer: X' or 'The answer is X' (case-insensitive)
      2. Trailing standalone letter on its own line
      3. Last standalone letter A–D in text
    """
    # Priority 1 — explicit label
    match = re.search(
        r'(?:answer(?:\s+is)?|therefore)[:\s]+\**([A-D])\b',
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Priority 2 — letter on its own line (common model pattern)
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip('*').strip('.')
        if line.upper() in LETTER_MAP:
            return line.upper()

    # Priority 3 — last bare A–D token
    tokens = re.findall(r'\b([A-D])\b', text.upper())
    return tokens[-1] if tokens else None


def extract_expected_letter(doc: dict) -> str | None:
    """
    lm-eval GSM8K uses 'answer'; MMLU uses integer index in 'gold'
    or a label in 'answer'. Handle both.
    """
    # lm-eval passes the numeric index in doc['gold'] for MMLU
    gold = doc.get("gold")
    if gold is not None:
        try:
            return IDX_MAP[int(gold)]
        except (KeyError, ValueError, TypeError):
            pass

    # fallback: 'answer' field as letter
    ans = doc.get("answer", "")
    if str(ans).upper() in LETTER_MAP:
        return str(ans).upper()

    # fallback: 'answer' as integer index
    try:
        return IDX_MAP[int(ans)]
    except (KeyError, ValueError, TypeError):
        pass

    return None


def ensure_mmlu_format(output: str) -> str:
    """
    Guarantee output ends with 'Answer: X' for lm-eval extraction.
    lm-eval MMLU uses loglikelihood by default, but since we override
    generate_until we need the letter cleanly surfaced.
    """
    letter = extract_choice_letter(output)
    if letter:
        # If already ends with a clean Answer line, leave it
        if re.search(r'Answer:\s*[A-D]\s*$', output.strip(), re.IGNORECASE):
            return output
        return output.strip() + f"\nAnswer: {letter}"
    return output


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FILE    = "irix_mmlu_per_question.csv"
CSV_HEADERS = [
    "q_index", "subject", "status",
    "expected_letter", "predicted_letter", "correct",
    "latency_ms", "route_path", "complexity",
    "used_search", "model_used", "answer_length",
    "prompt_tail", "raw_tail"
]

def init_csv() -> set[int]:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
        return set()
    completed = set()
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        completed = set(df["q_index"].dropna().astype(int).tolist())
        print(f"[Resume] {len(completed)} completed questions found.")
    except Exception as e:
        print(f"[Resume] Warning: {e}")
    return completed

def append_csv(row: dict):
    with open(CSV_FILE, "a", newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore').writerow(row)


# ---------------------------------------------------------------------------
# MMLU prompt injection
# ---------------------------------------------------------------------------

MMLU_SYSTEM_ADDON = (
    "MULTIPLE CHOICE RULE: When the question presents options (A, B, C, D), "
    "reason through them, then end your response with exactly one line: "
    "'Answer: X'  where X is the single letter. No parentheses, no extra text on that line."
)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@register_model("irix_mmlu")
class IrixMMLUWrapper(LM):
    def __init__(self):
        super().__init__()
        self.irix = IrixSystem()

        # Inject MCQ rule into the system prompt once
        base_sys = self.irix.history[0]["content"]
        if "MULTIPLE CHOICE RULE" not in base_sys:
            self.irix.history[0]["content"] = base_sys + "\n\n" + MMLU_SYSTEM_ADDON

        self.base_sys_prompt = self.irix.history[0]
        self.question_log: list[dict] = []
        self._q_index  = 0
        self._n_passed = 0
        self._completed = init_csv()

    def generate_until(self, requests):
        res = []
        for request in requests:
            self._q_index += 1
            prompt  = request.args[0]
            doc     = getattr(request, "doc", {})
            subject = doc.get("subject", "unknown")

            # ── RESUME ────────────────────────────────────────────────────
            if self._q_index in self._completed:
                print(f"[Q{self._q_index}] SKIPPED (already completed)")
                processed = self._load_processed_from_csv(self._q_index)
                if processed is None:
                    processed = "Answer: A"
                res.append(processed)
                letter = extract_choice_letter(processed)
                expected = extract_expected_letter(doc)
                if letter and letter == expected:
                    self._n_passed += 1
                continue

            # ── FRESH QUESTION ────────────────────────────────────────────
            expected_letter = extract_expected_letter(doc)
            self.irix.history = [self.base_sys_prompt]

            t0 = time.time()
            try:
                raw_output = self.irix.process(prompt)
            except Exception as e:
                print(f"[Q{self._q_index}] Error: {e}")
                raw_output = ""
            latency_ms = int((time.time() - t0) * 1000)

            raw_safe    = strip_emoji(raw_output)
            processed   = ensure_mmlu_format(raw_safe)
            predicted_l = extract_choice_letter(processed)

            passed = (predicted_l is not None and predicted_l == expected_letter)
            if passed:
                self._n_passed += 1

            status = "PASS" if passed else "FAIL"
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"[Q{self._q_index}] {status}  |  subject={subject}  "
                  f"expected={expected_letter}  predicted={predicted_l}  ({latency_ms}ms)")
            print(f"Score so far: {self._n_passed}/{self._q_index} "
                  f"({self._n_passed / self._q_index * 100:.1f}%)")
            print(f"PROMPT (last 200): ...{strip_emoji(prompt[-200:])}")
            print(f"RAW TAIL (last 300): ...{repr(raw_safe[-300:])}")
            print(f"{sep}\n")

            last_tel = self._read_last_telemetry()
            row = {
                "q_index":        self._q_index,
                "subject":        subject,
                "status":         status,
                "expected_letter": expected_letter,
                "predicted_letter": predicted_l,
                "correct":        passed,
                "latency_ms":     latency_ms,
                "route_path":     last_tel.get("router_output", {}).get("path", "?"),
                "complexity":     last_tel.get("router_output", {}).get("complexity", "?"),
                "used_search":    last_tel.get("used_search", False),
                "model_used":     last_tel.get("model_used", "?"),
                "answer_length":  len(raw_safe),
                "prompt_tail":    strip_emoji(prompt[-300:]),
                "raw_tail":       strip_emoji(raw_safe[-400:]),
            }
            self.question_log.append(row)
            append_csv(row)
            res.append(processed)
        return res

    def _load_processed_from_csv(self, q_index: int) -> str | None:
        try:
            df  = pd.read_csv(CSV_FILE, encoding='utf-8')
            row = df[df["q_index"] == q_index]
            if not row.empty:
                letter = row.iloc[0].get("predicted_letter")
                raw    = row.iloc[0].get("raw_tail", "")
                if isinstance(raw, str) and re.search(r'Answer:\s*[A-D]', raw, re.IGNORECASE):
                    return raw
                if letter and str(letter) in LETTER_MAP:
                    return f"Answer: {letter}"
        except Exception as e:
            print(f"[Resume] Warning Q{q_index}: {e}")
        return None

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
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IrixAI on MMLU")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific MMLU subjects (e.g. mmlu_anatomy). "
                             "Default: all 57 subjects via 'mmlu'.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of questions per subject (useful for smoke tests).")
    parser.add_argument("--fewshot", type=int, default=5,
                        help="Number of few-shot examples (default: 5, MMLU standard).")
    args = parser.parse_args()

    tasks   = args.subjects if args.subjects else ["mmlu"]
    wrapper = IrixMMLUWrapper()

    print(f"Starting MMLU eval | tasks={tasks} | limit={args.limit} | fewshot={args.fewshot}")

    results = simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=args.fewshot,
        limit=args.limit,
        random_seed=42
    )

    # ── Aggregate results ─────────────────────────────────────────────────
    skipped_indices = wrapper._completed
    new_rows        = wrapper.question_log

    if skipped_indices:
        try:
            df_existing  = pd.read_csv(CSV_FILE, encoding='utf-8')
            df_skipped   = df_existing[df_existing["q_index"].isin(skipped_indices)]
            df_questions = pd.concat(
                [df_skipped, pd.DataFrame(new_rows)], ignore_index=True
            ).sort_values("q_index")
        except Exception:
            df_questions = pd.DataFrame(new_rows)
    else:
        df_questions = pd.DataFrame(new_rows)

    total   = len(df_questions)
    correct = int(df_questions["correct"].sum()) if total else 0
    pass_rate = correct / total if total else 0.0

    # Per-subject breakdown
    if "subject" in df_questions.columns:
        df_by_subject = (
            df_questions.groupby("subject")["correct"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "correct", "count": "total"})
        )
        df_by_subject["accuracy"] = df_by_subject["correct"] / df_by_subject["total"]
        df_by_subject.to_csv("irix_mmlu_by_subject.csv", encoding='utf-8')
        print("\nPer-subject accuracy saved to irix_mmlu_by_subject.csv")

    agg_rows = []
    for task_name, metrics in results["results"].items():
        row = {"Task": task_name}
        row.update(metrics)
        row["irix_pass_rate"]  = round(pass_rate, 6)
        row["irix_pass_count"] = correct
        row["irix_fail_count"] = total - correct
        row["irix_total"]      = total
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_failures = df_questions[df_questions["correct"] == False]

    df_questions.to_csv("irix_mmlu_per_question.csv", index=False, encoding='utf-8')
    df_agg.to_csv(      "irix_mmlu_aggregate.csv",    index=False, encoding='utf-8')
    df_failures.to_csv( "irix_mmlu_failures.csv",     index=False, encoding='utf-8')

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"MMLU Evaluation complete:  {correct}/{total}  ({pass_rate * 100:.2f}%)")
    print(f"Outputs: per_question, aggregate, failures, by_subject CSVs")
    print(f"{sep}\n")
    print(utils.make_table(results))