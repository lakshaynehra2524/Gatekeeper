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



