from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.gatekeeper import GatekeeperEngine
from backend.schemas import EvaluateRequest, GateDecision, Scenario
from backend.simulator import load_scenarios, run_all

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"

app = FastAPI(
    title="Edge Gatekeeper",
    description="On-device gate that decides which conversational moments "
                "deserve deeper processing. Model assets: ~0.3 MB .",
    version="1.0.0",
)

# One engine per server process. In the real Android deployment this object is
# the on-device singleton fed by the speech-to-text stream.
engine = GatekeeperEngine()


# API 
@app.post("/api/evaluate", response_model=GateDecision, tags=["gatekeeper"])
def evaluate(req: EvaluateRequest):
    """Evaluate one incremental transcript fragment against the live session."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty utterance")
    return engine.evaluate(text).to_dict()


@app.post("/api/reset", tags=["gatekeeper"])
def reset():
    """Start a fresh conversation session (clears context, dedup memory, stats)."""
    engine.reset()
    return {"status": "reset"}


@app.get("/api/model-info", tags=["gatekeeper"])
def model_info():
    """Model metadata, size , thresholds and session stats."""
    return engine.model_info()


@app.get("/api/scenarios", response_model=list[Scenario], tags=["simulation"])
def scenarios():
    """Hand-written replay conversations used by the demo and the evaluation."""
    return load_scenarios()


@app.get("/api/run-eval", tags=["simulation"])
def run_eval():
    """Replay every scenario on a fresh engine and score against expectations."""
    return run_all(GatekeeperEngine())



