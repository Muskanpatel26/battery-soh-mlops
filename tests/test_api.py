import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

from src.api import app


ROOT = (
    Path(__file__).resolve().parents[1]
)


def test_model_and_api():
    metadata = json.loads(
        (
            ROOT /
            "bundle/metadata.json"
        ).read_text()
    )

    assert (
        metadata["release_gate"]
        == "passed"
    )

    train = set(
        metadata["split"]["train"]
    )

    validation = set(
        metadata["split"]["validation"]
    )

    test = set(
        metadata["split"]["test"]
    )

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    rows = json.loads(
        (
            ROOT /
            "bundle/sample_input.json"
        ).read_text()
    )

    model = mlflow.sklearn.load_model(
        str(
            ROOT / "bundle/model"
        )
    )

    frame = pd.DataFrame(
        rows
    )[metadata["features"]].astype(
        "float64"
    )

    expected = model.predict(frame)

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        assert (
            response.json()["model_version"]
            == metadata["model_version"]
        )

        response = client.post(
            "/predict",
            json={
                "instances": rows
            }
        )

        assert (
            response.status_code == 200
        ), response.text

        np.testing.assert_allclose(
            response.json()[
                "predicted_SOH_percent"
            ],
            expected,
            rtol=1e-6
        )

        assert client.post(
            "/predict",
            json={
                "instances": []
            }
        ).status_code == 422

        assert client.post(
            "/predict",
            json={
                "instances": [
                    {"cycle": 1}
                ]
            }
        ).status_code == 422
