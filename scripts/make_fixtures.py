"""Regenerate the frozen synthetic fixture corpus in evals/fixtures/histories/.

    python scripts/make_fixtures.py

Fixtures are committed so evals run against a fixed corpus; tests/test_fixtures.py checks
they still match the generator.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spa.synth import PERSONAS, TRUTH, generate  # noqa: E402

SEEDS = {p: list(range(5)) for p in PERSONAS}
SEEDS["null"] = list(range(10))
SEEDS["sparse"] = [0, 1]


def main() -> None:
    out = ROOT / "evals" / "fixtures" / "histories"
    out.mkdir(parents=True, exist_ok=True)
    for persona, seeds in SEEDS.items():
        doc = {
            "persona": persona,
            "description": PERSONAS[persona],
            "truth": TRUTH[persona],
            "synthetic": True,
            "histories": {
                str(s): [e.model_dump(mode="json", exclude_defaults=True) for e in generate(persona, s)] for s in seeds
            },
        }
        (out / f"{persona}.json").write_text(json.dumps(doc, separators=(",", ":")))
        print(f"wrote {persona}: seeds {seeds}")


if __name__ == "__main__":
    main()
