"""Depth Dive route: POST /dive (depth-dive spec §9)."""

from fastapi import APIRouter, HTTPException

from core.types.depth_dive import HarnessBRequest, HarnessBResponse
from depth_dive.harness import run_dive
from depth_dive.transform import PassageTransformError

router = APIRouter(tags=["dive"])


@router.post("/dive", response_model=HarnessBResponse)
def dive(body: HarnessBRequest) -> HarnessBResponse:
    """Generate a Depth Dive interactive animation for a Captured Passage.

    Returns 200 with a ``HarnessBResponse`` carrying the hardcoded demo
    ``interactive_animation`` scene graph. Passages that violate the declared
    size/bounds or that the tracer bullet does not support are rejected with
    422 (``PassageTransformError``). Malformed request bodies return 422 via
    FastAPI defaults.
    """
    try:
        return run_dive(body.captured_passage)
    except PassageTransformError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = ["router"]
