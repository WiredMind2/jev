from jev.data.synthetic import make_synthetic_choice
from jev.scoring.option_head import TrainConfig, accuracy_on_examples, train_option_head


def test_hashing_head_learns_synthetic_and_uses_state() -> None:
    rows = make_synthetic_choice(n=80, k=4, seed=0)
    encoder, head, history = train_option_head(
        rows, TrainConfig(epochs=12, batch_size=8, seed=0, lr=3e-3)
    )
    acc = accuracy_on_examples(encoder, head, rows)
    shuf = accuracy_on_examples(encoder, head, rows, shuffle_state=True, seed=3)
    assert acc >= 0.9
    assert shuf < acc - 0.3
    assert history["final_loss"] >= 0.0


def test_early_stopping_uses_validation_not_calibration() -> None:
    rows = make_synthetic_choice(n=64, seed=1)
    train, val = rows[:48], rows[48:]
    encoder, head, hist = train_option_head(
        train,
        TrainConfig(epochs=12, patience=4, batch_size=8, seed=1, lr=3e-3),
        val_examples=val,
    )
    assert "best_val_acc" in hist
    acc = accuracy_on_examples(encoder, head, train)
    assert acc >= 0.5
