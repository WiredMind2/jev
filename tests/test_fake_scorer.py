from jev.pipeline import respond
from jev.schema import SystemOneRequest
from jev.scoring.fake import FakeScorer, overlap_logit, tokenize


def test_fake_scorer_prefers_overlapping_option() -> None:
    req = SystemOneRequest(
        state="The Stripe integration keeps failing and sales are down.",
        questions={
            "department": {
                "type": "choice",
                "instructions": "route",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions",
                },
            }
        },
    )
    resp = respond(FakeScorer(), req)
    ans = resp.answers["department"]
    assert ans.type == "choice"
    assert ans.choice == "technical"
    assert abs(sum(ans.probabilities.values()) - 1.0) < 1e-6


def test_fake_overlap_is_deterministic() -> None:
    tokens = tokenize("refund refund please")
    a = overlap_logit(tokens, "refund request")
    b = overlap_logit(tokens, "refund request")
    c = overlap_logit(tokens, "sales pricing")
    assert a == b
    assert a > c
