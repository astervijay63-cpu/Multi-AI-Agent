# PURPOSE:
#   Handles POST /api/v1/analyze — the most important endpoint.
#   This is the endpoint that triggers a full MOSAIC analysis run.

# WHAT HAPPENS WHEN THIS ENDPOINT IS CALLED:
#   1. Receives the analysis task and parameters
#   2. Invokes the compiled LangGraph graph with the task
#   3. The graph runs all 6 specialist agents in parallel
#   4. Signals are processed through the HITL gate
#   5. Supervisor compiles the final intelligence brief
#   6. Returns the complete analysis result to the caller

# THIS IS THE ENTRY POINT FOR EVERYTHING:
#   All the infrastructure we built — ingestion, processing,
#   memory, agents, tools — is exercised by this one endpoint.


import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import AnalysisRequest, AnalysisResponse
from api.dependencies import get_hitl_gate, get_graph
from config.settings import settings
from graph.hitl import HITLGate
from config.logging_config import setup_logging

logger = setup_logging(__name__)

router = APIRouter()



@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Trigger a full MOSAIC analysis run",
    description="Runs all 6 specialist agents in parallel and returns "
                "a compiled intelligence brief with all signals found.",
    tags=["Analysis"],

)
async def run_analysis(
    request: AnalysisRequest,
    hitl_gate: HITLGate = Depends(get_hitl_gate),
 
):
    """
    Triggers a complete MOSAIC analysis run.

    Runs all 6 specialist agents in parallel:
    - Missing Results Agent
    - Broken Promises Agent
    - Track Record Agent
    - Pattern Finder Agent
    - Side Effect Checker
    - Timeline Analyst

    Returns a compiled intelligence brief with all signals found.
    """

    run_id     = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Analysis run started | "
        f"run_id={run_id} | "
        f"task='{request.task[:80]}'"
    )

    try:
        if not settings.google_api_key:
            raise HTTPException(
                status_code=503,
                detail="Analysis requires GOOGLE_API_KEY to be configured in .env.",
            )

        graph = get_graph()
        initial_state = {
            "task":        request.task,
            "nct_ids":     request.nct_ids,
            "max_studies": request.max_studies,
            "messages":    [],
            "signals":     [],
            "agents_activated": [],
            "final_brief": "",
            "run_complete": False,
            "run_id":      run_id,
            "error_log":   [],
        }

        result = await graph.ainvoke(initial_state)

        signals          = result.get("signals", [])
        processed_signals = []

        persistence_failures = 0

        for signal in signals:
            # Persist each signal independently. A single bad one used to
            # propagate out of this loop and 500 the whole request, which
            # threw away the brief and every other agent's findings — the
            # expensive part of the run — over one unstorable row.
            try:
                hitl_result = await hitl_gate.process_signal(signal)
                action      = hitl_result.get("action")

            except Exception as e:
                persistence_failures += 1
                action = "persist_failed"
                logger.error(
                    f"Signal could not be persisted | "
                    f"run_id={run_id} | "
                    f"agent={signal.get('agent')} | "
                    f"nct_id={signal.get('nct_id')!r} | "
                    f"error={e}"
                )

            processed_signals.append({
                **signal,
                # ** unpacks the signal dict into the new dict.
                "hitl_action": action,
            })

        duration = round(time.time() - start_time, 2)

        signals_requiring_review = sum(
            1 for s in processed_signals
            if s.get("hitl_action") == "sent_to_review"
        )
        logger.info(
            f"Analysis run complete | "
            f"run_id={run_id} | "
            f"signals={len(signals)} | "
            f"persistence_failures={persistence_failures} | "
            f"duration={duration}s"
        )

        return AnalysisResponse(
            run_id=run_id,
            task=request.task,
            final_brief=result.get("final_brief", "No brief generated."),
            total_signals=len(signals),
            signals_requiring_review=signals_requiring_review,
            agents_activated=result.get("agents_activated", []),
            duration_seconds=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis run failed | run_id={run_id} | error={e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis run failed: {str(e)}",
        )
