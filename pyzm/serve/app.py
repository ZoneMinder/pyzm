"""FastAPI application factory for the pyzm ML detection server."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from pyzm.ml.detector import Detector
from pyzm.models.config import ServerConfig
from pyzm.models.zm import Zone
from pyzm.serve.auth import create_login_route, create_token_dependency

logger = logging.getLogger("pyzm.serve")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    The :class:`Detector` is created during the lifespan startup phase so
    models are loaded once and persist across requests.
    """
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        detector = Detector(
            models=config.models,
            base_path=config.base_path,
            processor=config.processor,
        )
        # Eagerly load models so the first request doesn't pay the cost
        detector._ensure_pipeline()
        app.state.detector = detector
        logger.info(
            "Detector ready: %d model(s) loaded", len(detector._config.models)
        )
        yield

    app = FastAPI(title="pyzm ML Detection Server", lifespan=lifespan)

    # -- Optional auth -------------------------------------------------------
    auth_deps: list[Any] = []
    if config.auth_enabled:
        verify_token = create_token_dependency(config)
        auth_deps = [Depends(verify_token)]
        app.post("/login")(create_login_route(config))

    # -- Routes --------------------------------------------------------------

    @app.get("/health")
    def health():
        models_loaded = (
            hasattr(app.state, "detector") and app.state.detector._pipeline is not None
        )
        return {"status": "ok", "models_loaded": models_loaded}

    @app.post("/detect", dependencies=auth_deps)
    async def detect(
        file: UploadFile = File(...),
        zones: str | None = Form(None),
    ):
        import cv2
        import numpy as np

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        zone_list = None
        if zones:
            try:
                raw_zones = json.loads(zones)
                zone_list = [
                    Zone(
                        name=z.get("name", ""),
                        points=z.get("value", z.get("points", [])),
                        pattern=z.get("pattern"),
                    )
                    for z in raw_zones
                ]
            except (json.JSONDecodeError, TypeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid zones JSON: {exc}"
                )

        detector: Detector = app.state.detector
        result = detector.detect(image, zones=zone_list)

        data = result.to_dict()
        # Strip non-serializable fields
        data.pop("image", None)
        return data

    return app
