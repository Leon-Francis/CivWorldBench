"""Use the established engine loop with an isolated deliberative agent adapter."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.agentplay import run_game
from bench.agentplay.deliberative import DeliberativeAgent


def clean_siege(lines):
    result = []
    for line in lines:
        if "Measured base rate" in line:
            result.append(line.split("Measured base rate")[0].rstrip(" .*★") + ".")
        elif any(part in line for part in ("within the next 20 turns in only", "0.7) -- most stacks", "'this city will fall'")):
            continue
        else:
            result.append(line)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--turns", type=int, default=150)
    parser.add_argument("--score_end", type=int)
    parser.add_argument("--wm_model", default="")
    parser.add_argument("--wm_endpoint", default="http://localhost:8011/v1")
    args = parser.parse_args()
    DeliberativeAgent.config = {**vars(args), "ego": 0, "end": args.score_end or args.start + args.turns}
    original_view = run_game.build_view

    def view(*positional, **keyword):
        observed = original_view(*positional, **keyword)
        observed["rules"] = positional[3] if len(positional) > 3 else keyword["rules"]
        old = "\n".join(observed.get("siege") or [])
        new = "\n".join(clean_siege(observed.get("siege") or []))
        if old:
            observed["panel"] = observed["panel"].replace(old, new)
        observed["siege"] = new.splitlines()
        DeliberativeAgent.current_view = observed
        return observed

    run_game.Agent = DeliberativeAgent
    run_game.build_view = view
    sys.argv = [sys.argv[0], "--world", args.world, "--out", args.out, "--start", str(args.start),
                "--turns", str(args.turns), "--arm", "none", "--agent_naive", "1"]
    failure_path = Path(args.out) / "failure.json"
    failure_path.unlink(missing_ok=True)
    try:
        run_game.main()
    except Exception as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps({"error": str(error), "type": type(error).__name__,
            "retryable": isinstance(error, RuntimeError) and str(error).startswith(
                ("Invalid planning JSON", "inference failed:"))}) + "\n")
        raise
    summary_path = Path(args.out) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(arm="deliberative", wm_model=args.wm_model or None, protocol="deliberative-v1-r2")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
