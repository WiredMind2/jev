from jev.render import continuations, option_keys, render_prefix
from jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


def test_render_includes_state_and_options() -> None:
    q = ChoiceQuestion(
        type="choice",
        instructions="Which bucket?",
        criteria={"azure": "blue", "ochre": "yellow"},
    )
    prefix = render_prefix({"ticket": "azure"}, q)
    assert "azure" in prefix
    assert "ochre" in prefix
    assert option_keys(q) == ["azure", "ochre"]
    assert continuations(q) == [" azure", " ochre"]


def test_score_and_noul_continuations() -> None:
    s = ScoreQuestion(type="score", instructions="mood", criteria=["calm", "angry"])
    n = NoulQuestion(type="noul", instructions="urgent?")
    assert continuations(s) == [" 0", " 1"]
    assert continuations(n) == [" false", " true"]
