from jev.hardware import DEFAULT_PIN, load_hardware_pin


def test_hardware_pin_qwen25_0_5b() -> None:
    pin = load_hardware_pin()
    assert pin.zero_shot_model == "Qwen/Qwen2.5-0.5B"
    assert pin.frozen_head_model == "Qwen/Qwen2.5-0.5B"
    assert pin.vram_mib == 4096
    assert pin.fp16 is True
    assert pin.bf16 is False
    assert "3B" in pin.reduction_reason or "3b" in pin.reduction_reason.lower()
    assert DEFAULT_PIN.zero_shot_model == pin.zero_shot_model
