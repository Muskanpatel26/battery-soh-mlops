import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException

from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)


# Resolve paths relative to this project,
# rather than the terminal's working directory.
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

BUNDLE_DIRECTORY = (
    PROJECT_ROOT / "bundle"
)

METADATA = json.loads(
    (
        BUNDLE_DIRECTORY /
        "metadata.json"
    ).read_text()
)

FEATURES = METADATA["features"]

MODEL = mlflow.sklearn.load_model(
    str(
        BUNDLE_DIRECTORY / "model"
    )
)

app = FastAPI(
    title="Battery SOH Prediction API",
    version="1.0"
)


class BatteryInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    cycle: float = Field(ge=0)

    ambient_temperature: float

    voltage_mean: float
    voltage_min: float
    voltage_max: float

    voltage_std: float = Field(ge=0)

    current_mean: float
    current_std: float = Field(ge=0)

    temperature_mean: float
    temperature_max: float


class PredictionRequest(BaseModel):
    instances: list[BatteryInput] = Field(
        min_length=1,
        max_length=500
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name":
            METADATA["model_name"],
        "model_version":
            METADATA["model_version"]
    }


@app.get("/metadata")
def metadata():
    return METADATA


@app.post("/predict")
def predict(request: PredictionRequest):
    rows = [
        instance.model_dump()
        for instance in request.instances
    ]

    frame = pd.DataFrame(
        rows
    )[FEATURES].astype("float64")

    predictions = MODEL.predict(frame)

    if not np.isfinite(
        predictions
    ).all():
        raise HTTPException(
            status_code=500,
            detail="Non-finite model output."
        )

    return {
        "model_name":
            METADATA["model_name"],
        "model_version":
            METADATA["model_version"],
        "run_id":
            METADATA["run_id"],
        "predicted_SOH_percent": [
            float(value)
            for value in predictions
        ]
    }
