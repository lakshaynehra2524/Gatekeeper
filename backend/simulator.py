"""
Conversation simulator.

Replays hand-written conversations from data/eval_scenarios.json as if the
fragments were arriving live from the speech-to-text layer, and scores the
Gatekeeper's decisions against the expected behaviour.

"""
from __future__ import annotations

import json
import time
from pathlib import Path

from backend.gatekeeper import GatekeeperEngine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_scenarios() -> list[dict]:
    with open(DATA_DIR / "eval_scenarios.json") as f:
        return json.load(f)


def run_scenario(engine: GatekeeperEngine, scenario: dict) -> dict:
    """Replay one scenario on a fresh session; return turn-by-turn results."""
    engine.reset()
    turns = []
    correct = 0
    for turn in scenario["turns"]:
        result = engine.evaluate(turn["text"]).to_dict()
        # UNCERTAIN forwarded-with-flag counts as matching a FORWARD expectation
        # only when the expectation itself is UNCERTAIN; strict otherwise.
        match = result["decision"] == turn["expected"]
        correct += match
        turns.append({**result, "expected": turn["expected"], "match": match})
    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "turns": turns,
        "correct": correct,
        "total": len(turns),
    }


def run_all(engine: GatekeeperEngine) -> dict:
    reports = [run_scenario(engine, s) for s in load_scenarios()]
    return {
        "scenarios": reports,
        "correct": sum(r["correct"] for r in reports),
        "total": sum(r["total"] for r in reports),
    }


# CLI demo 
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Edge Gatekeeper simulator")
    parser.add_argument("--live", action="store_true",
                        help="interactive mode: type utterances yourself")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="seconds between replayed fragments")
    args = parser.parse_args()

    engine = GatekeeperEngine()
    icons = {"FORWARD": "->", "REJECT": "x ", "UNCERTAIN": "??", "DUPLICATE": "= "}

    if args.live:
        print("Live mode — type an utterance (empty line to quit)\n")
        while True:
            try:
                text = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            r = engine.evaluate(text)
            print(f"  [{icons[r.decision]}] {r.decision:9s} label={r.label:9s} "
                  f"p_meaningful={r.p_meaningful:.2f}  ({r.latency_ms:.1f} ms)")
            print(f"       {r.reason}")
        return

    for scenario in load_scenarios():
        print(f"\n=== {scenario['name']} — {scenario['description']}")
        engine.reset()
        for turn in scenario["turns"]:
            time.sleep(args.delay)
            r = engine.evaluate(turn["text"])
            ok = "OK " if r.decision == turn["expected"] else "MISS"
            print(f" [{icons[r.decision]}] {r.decision:9s} (want {turn['expected']:9s}) "
                  f"{ok}  \"{turn['text']}\"  p={r.p_meaningful:.2f} "
                  f"{r.latency_ms:.1f}ms")
    report = run_all(GatekeeperEngine())
    print(f"\nOverall: {report['correct']}/{report['total']} turns matched expectation")


if __name__ == "__main__":
    _cli()
    
