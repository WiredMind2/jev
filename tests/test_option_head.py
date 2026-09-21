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


def test_resume_continues_from_saved_step(tmp_path) -> None:
    rows = make_synthetic_choice(n=48, seed=2)
    train, val = rows[:36], rows[36:]
    ckpt = tmp_path / "mid.pt"
    _, _, first = train_option_head(
        train,
        TrainConfig(
            epochs=4,
            batch_size=8,
            seed=2,
            max_steps=3,
            save_every=1,
            checkpoint_path=ckpt,
        ),
        val_examples=val,
    )
    assert ckpt.exists()
    assert first["step"] == 3.0
    _, _, second = train_option_head(
        train,
        TrainConfig(
            epochs=4,
            batch_size=8,
            seed=2,
            max_steps=6,
            save_every=1,
            checkpoint_path=ckpt,
            resume_path=ckpt,
        ),
        val_examples=val,
    )
    assert second["resumed_step"] == 3.0
    assert second["step"] == 6.0
    assert second["resumed_step"] < second["step"]


def test_resume_at_max_steps_does_not_train_further(tmp_path) -> None:
    from jev.scoring.option_head import checkpoint_train_step

    rows = make_synthetic_choice(n=48, seed=2)
    train = rows[:36]
    ckpt = tmp_path / "done.pt"
    _, _, first = train_option_head(
        train,
        TrainConfig(
            epochs=4,
            batch_size=8,
            seed=2,
            max_steps=3,
            save_every=1,
            checkpoint_path=ckpt,
        ),
    )
    assert first["step"] == 3.0
    assert checkpoint_train_step(ckpt) == 3
    _, _, second = train_option_head(
        train,
        TrainConfig(
            epochs=4,
            batch_size=8,
            seed=2,
            max_steps=3,
            save_every=1,
            checkpoint_path=ckpt,
            resume_path=ckpt,
        ),
    )
    assert second["resumed_step"] == 3.0
    assert second["step"] == 3.0
