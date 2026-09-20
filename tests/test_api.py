import json

from fastapi.testclient import TestClient

from jev.api import create_app
from jev.scoring.fake import FakeScorer

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_health_and_models() -> None:
    client = TestClient(create_app(FakeScorer()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready"
    models = client.get("/v1/models").json()
    assert models["models"]
    assert models["hardware"]["zero_shot_model"] == "Qwen/Qwen2.5-0.5B"
    assert models["hardware"]["frozen_head_model"] == "Qwen/Qwen2.5-0.5B"


def test_systemone_choice_score_noul() -> None:
    payload = json.loads((REPO_ROOT / "examples" / "systemone.request.json").read_text())
    client = TestClient(create_app(FakeScorer()))
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    answers = body["answers"]
    assert answers["department"]["type"] == "choice"
    assert answers["department"]["choice"] in answers["department"]["probabilities"]
    assert abs(sum(answers["department"]["probabilities"].values()) - 1.0) < 1e-5
    assert answers["frustration"]["type"] == "score"
    assert "confidence" in answers["frustration"]
    assert answers["is_urgent"]["type"] == "noul"
    assert "confidence" not in answers["is_urgent"]
    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0


def test_systemone_rejects_one_option() -> None:
    client = TestClient(create_app(FakeScorer()))
    resp = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "q": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}
            },
        },
    )
    assert resp.status_code == 422
