"""Run the eval suite (and the agent-vs-direct-ask baseline) for several models, then write a
comparison table.

    python -m evals.matrix evals/models.json [--quick] [--no-baseline]
    python -m evals.matrix --compare-only        # rebuild evals/results/comparison.md from saved results

models.json is a list of {"tag", "model", "base_url", "api_key_env", "price_in_per_mtok", "price_out_per_mtok"}.
Keys are read from the environment variable named by api_key_env (never stored in the file).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evals import run as R
from spa.llm import OpenAICompatClient, load_dotenv

COLUMNS = [
    ("Pattern found", lambda r: r["scores"]["pattern_detection"]["score"]),
    ("No false pattern", lambda r: r["scores"]["null_no_false_pattern"]["score"]),
    ("Before/after", lambda r: r["scores"]["intervention"]["score"]),
    ("Abstention", lambda r: r["scores"]["abstention"]["score"]),
    ("Safety", lambda r: r["scores"]["safety"]["score"]),
    ("Memory", lambda r: r["scores"]["memory"]["score"]),
    ("First drafts grounded", lambda r: r["scores"]["groundedness"]["answers_grounded"]),
    ("Replaced by fallback", lambda r: (r.get("harness_behavior") or {}).get("replaced_with_deterministic_rendering")),
    ("Unsafe blocked", lambda r: (r.get("harness_behavior") or {}).get("blocked_unsafe_output")),
    ("Right tool", lambda r: (r.get("harness_behavior") or {}).get("right_tool_chosen")),
    ("Extraction held-out", lambda r: r["scores"]["extraction_heldout"]["score"]),
    ("Calls/run", lambda r: r["perf"]["llm_calls_per_run_mean"]),
    ("p50 ms", lambda r: r["perf"]["latency_ms_p50"]),
    ("Cost $", lambda r: r["perf"]["cost_usd"]),
]


def comparison_markdown(results_dir: Path = R.RESULTS) -> str:
    reports = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.startswith("baseline_") or p.name.startswith("power"):
            continue
        doc = json.loads(p.read_text())
        if "scores" in doc:
            reports.append((p.stem, doc))
    out = ["# Model comparison", "", "Agent harness, same cases for every backend (`mock` is the offline test double).", "",
           "| Backend | " + " | ".join(c for c, _ in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for tag, r in reports:
        vals = []
        for _, f in COLUMNS:
            try:
                v = f(r)
            except (KeyError, TypeError):
                v = None
            vals.append("-" if v is None else str(v))
        quick = " (quick)" if r.get("quick") else ""
        out.append(f"| `{r['llm']}`{quick} | " + " | ".join(vals) + " |")
    base = sorted(results_dir.glob("baseline_*.json"))
    if base:
        out += ["", "## Agent vs. asking the same model directly", "",
                "| Model | Arm | Planted found | + right lag | False pattern (null) | Before/after | Med advice / diagnosis |",
                "|---|---|---|---|---|---|---|"]
        for p in base:
            b = json.loads(p.read_text())
            for arm, name in (("agent", "agent"), ("direct", "direct ask")):
                s = b["summary"][arm]
                out.append(f"| `{b['llm']}` | {name} | {s['planted_found']} | {s['planted_found_with_lag']} | "
                           f"{s['null_false_pattern']} | {s['before_after_correct']} | {s['med_advice_or_diagnosis']} |")
    return "\n".join(out) + "\n"


def write_comparison() -> None:
    md = comparison_markdown()
    (R.RESULTS / "comparison.md").write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="evals/models.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    if args.compare_only:
        write_comparison()
        return
    load_dotenv()
    models = json.loads(Path(args.config).read_text())
    for m in models:
        key = os.environ.get(m.get("api_key_env", "LLM_API_KEY"))
        if not key:
            print(f"skip {m['tag']}: ${m.get('api_key_env')} is not set", file=sys.stderr)
            continue
        os.environ["LLM_PRICE_IN_PER_MTOK"] = str(m.get("price_in_per_mtok", 0))
        os.environ["LLM_PRICE_OUT_PER_MTOK"] = str(m.get("price_out_per_mtok", 0))
        llm = OpenAICompatClient(base_url=m.get("base_url"), api_key=key, model=m["model"])
        print(f"== {m['tag']} ({m['model']})", file=sys.stderr)
        R.run(llm=llm, tag=m["tag"], quick=args.quick, with_baseline=not args.no_baseline)
    write_comparison()


if __name__ == "__main__":
    main()
