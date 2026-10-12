"""Record immutable question/source hashes before the formal experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cot-v32-190")
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    verification = json.loads((run / "verification.json").read_text())
    if not verification["ok"]:
        raise ValueError("Unverified question set")
    files = sorted((run / "questions").glob("*.jsonl.gz"))
    manifest = {
        "protocol": "3.2-aligned", "source_run": "v3-bench-br", "verification": verification,
        "selection_sha256": digest(run / "selection.json"),
        "question_digest": hashlib.sha256(b"".join(bytes.fromhex(digest(p)) for p in files)).hexdigest(),
        "questions": {p.name: digest(p) for p in files},
        "frozen_source_questions": {p.name: digest(ROOT / "bench/runs/v3-bench-br/questions" / p.name) for p in files},
        "code_at_freeze": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "bench/cot").glob("*.py"))},
        "assignment": {"node111": ["q38-27b", "kimi-k3"], "node108": ["gemma4-31b", "flash-next"]},
        "hypotheses": ["Answer skill need not imply accurate intermediate forecasts.",
                       "Code may trade process accuracy against final-answer quality and cost.",
                       "Within-model process-answer association may be stronger for paired L3 effects than matched L1 states."],
        "decision_policy": "Report negative or uncertain findings; no score-dependent pilot approval or anchor selection.",
    }
    path = run / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to overwrite a different frozen manifest")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"anchors": len(files), "question_digest": manifest["question_digest"]}))


if __name__ == "__main__":
    main()
