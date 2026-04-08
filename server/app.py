"""
FastAPI server for the ClinicalTrialEnv OpenEnv environment.

Endpoints (OpenEnv spec compliant):
  GET  /health      -> {"status": "healthy"}
  GET  /metadata    -> name, description, version, tasks
  GET  /schema      -> action, observation, state schemas
  GET  /tasks       -> list tasks with graders
  POST /reset       -> reset episode, return initial observation
  POST /step        -> take action, return step result
  GET  /state       -> return current internal state
  POST /mcp         -> JSON-RPC 2.0 endpoint (OpenEnv runtime check)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from environment import ClinicalTrialEnv
from models import ClinicalTrialAction
from tasks import TASKS

app = FastAPI(
    title="ClinicalTrialEnv",
    description="OpenEnv environment for Clinical Trial Protocol Review.",
    version="1.0.0",
)
@app.get("/")
def root():
    return {"status": "healthy", "environment": "ClinicalTrialEnv", "version": "1.0.0"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[ClinicalTrialEnv] = None


class ResetRequest(BaseModel):
    task: str = "eligibility_screening"

class StepRequest(BaseModel):
    findings: List[Dict[str, Any]] = []
    rationale: str = ""

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# OpenEnv required runtime endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """OpenEnv spec: must return {"status": "healthy"}"""
    return {"status": "healthy", "environment": "ClinicalTrialEnv", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    """OpenEnv spec: must return name and description."""
    return {
        "name": "clinical-trial-env",
        "description": (
            "OpenEnv environment for Clinical Trial Protocol Review. "
            "AI agents act as medical monitors — screening patients for eligibility "
            "violations, classifying adverse event severity (CTCAE), and reviewing "
            "protocol amendments for GCP/ICH compliance."
        ),
        "version": "1.0.0",
        "tasks": [
            {
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
                "has_grader": True,
                "score_range": [0.0, 1.0],
            }
            for t in TASKS.values()
        ],
    }


@app.get("/schema")
def schema():
    """OpenEnv spec: must return action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "description": "Structured clinical review findings",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding_type": {
                                "type": "string",
                                "enum": [
                                    "protocol_deviation",
                                    "adverse_event",
                                    "eligibility_violation",
                                    "safety_concern",
                                    "amendment_recommendation",
                                ],
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor", "informational"],
                            },
                            "subject_id": {"type": "string", "nullable": True},
                            "description": {"type": "string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["finding_type", "severity", "description", "recommendation"],
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["findings", "rationale"],
        },
        "observation": {
            "type": "object",
            "description": "Clinical trial data presented to the agent",
            "properties": {
                "task_name": {"type": "string"},
                "protocol_summary": {"type": "string"},
                "patient_records": {"type": "array"},
                "adverse_events": {"type": "array"},
                "protocol_text": {"type": "string"},
                "step": {"type": "integer"},
                "feedback": {"type": "string"},
                "partial_score": {"type": "number"},
            },
        },
        "state": {
            "type": "object",
            "description": "Internal environment state",
            "properties": {
                "task_name": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "current_score": {"type": "number"},
                "n_findings_accumulated": {"type": "integer"},
                "history": {"type": "array"},
                "elapsed_seconds": {"type": "number"},
            },
        },
    }


@app.post("/mcp")
def mcp(payload: Dict[str, Any] = {}):
    """JSON-RPC 2.0 endpoint required by OpenEnv runtime validator."""
    method = payload.get("method", "")
    req_id = payload.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "clinical-trial-env", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment to a new task episode",
                        "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}}},
                    },
                    {
                        "name": "step",
                        "description": "Submit findings and advance the episode",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "findings": {"type": "array"},
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                ]
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "environment": "clinical-trial-env",
            "version": "1.0.0",
            "tasks": list(TASKS.keys()),
        },
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
                "has_grader": True,
                "score_range": [0.0, 1.0],
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest = None) -> ResetResponse:
    global _env
    if req is None:
        req = ResetRequest()
    task = req.task if req and req.task else "eligibility_screening"
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Available: {list(TASKS.keys())}")
    _env = ClinicalTrialEnv(task_name=task)
    result = _env.reset()
    return ResetResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    action = ClinicalTrialAction(findings=req.findings, rationale=req.rationale)
    result = _env.step(action)
    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    global _env
    if _env is None:
        return {"status": "not_initialized"}
    return _env.state()


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()