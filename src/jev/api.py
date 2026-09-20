"""FastAPI System One server. Policy/authorization stays outside."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from jev import __version__
from jev.hardware import load_hardware_pin
from jev.pipeline import respond
from jev.schema import SystemOneRequest
from jev.scoring.fake import FakeScorer
from jev.scoring.protocol import Scorer


def create_app(scorer: Scorer | None = None) -> FastAPI:
    app = FastAPI(title="open jev-like research v0", version=__version__)
    app.state.scorer = scorer or FakeScorer()
    pin = load_hardware_pin()

    @app.get("/health")
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ready", "model": app.state.scorer.model_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": app.state.scorer.model_id,
                    "description": "Local research scorer. Not TypeSafe Jev.",
                    "release_date": "2026-09-20",
                }
            ],
            "hardware": pin.as_dict(),
        }

    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any]) -> Any:
        try:
            request = SystemOneRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
        try:
            response = respond(app.state.scorer, request)
        except Exception as exc:  # pragma: no cover - mapped below
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(response.model_dump(mode="json"))

    return app


app = create_app()
