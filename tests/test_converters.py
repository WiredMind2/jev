from pathlib import Path

from jev.data.banking77 import convert_banking77
from jev.data.boolq import convert_boolq
from jev.data.clinc150 import convert_clinc150
from jev.data.convert import freeze_and_write
from jev.data.manifest import criteria_path, load_criteria
from jev.data.splits import assert_no_group_overlap
from jev.data.sst5 import convert_sst5
from jev.data.synthetic import make_synthetic_choice, split_synthetic
from jev.data.wikispeedia import convert_wikispeedia, replay_path, sample_menu
from jev.schema import parse_training_example

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _read_split(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(parse_training_example(__import__("json").loads(line)))
    return rows


def test_synthetic_group_splits(tmp_path: Path) -> None:
    rows = make_synthetic_choice(n=64, seed=0)
    splits = split_synthetic(rows, seed=0)
    assert set(splits) == {"train", "validation", "calibration", "test"}
    assert all(splits[name] for name in splits)
    freeze_and_write(
        dataset="synthetic",
        primitive="choice",
        criteria_file=criteria_path("synthetic"),
        converter="test",
        license_name="generated",
        source="test",
        split_rule="grouped",
        examples_by_split=splits,
        out_dir=tmp_path,
    )
    loaded = {name: _read_split(tmp_path / "jsonl" / f"{name}.jsonl") for name in splits}
    assert_no_group_overlap(loaded)


def test_all_fixture_converters_emit_four_nonempty_splits(tmp_path: Path) -> None:
    """Fixtures must be large enough for train/val/calib/test, not empty carved splits."""
    jobs = (
        ("banking77", convert_banking77),
        ("sst5", convert_sst5),
        ("boolq", convert_boolq),
        ("clinc150", convert_clinc150),
        ("wikispeedia", convert_wikispeedia),
    )
    names = ("train", "validation", "calibration", "test")
    for dataset, fn in jobs:
        dest = tmp_path / dataset
        dest.mkdir()
        fn(dest, fixture=FIXTURES / "data" / f"{dataset}.json")
        splits = {name: _read_split(dest / "jsonl" / f"{name}.jsonl") for name in names}
        assert all(splits[name] for name in names), {k: len(v) for k, v in splits.items()}
        assert_no_group_overlap(splits)
        manifest = __import__("json").loads((dest / "manifest.json").read_text())
        assert set(manifest["splits"]) >= set(names)


def test_banking77_does_not_use_intent_as_group_id(tmp_path: Path) -> None:
    convert_banking77(tmp_path, fixture=FIXTURES / "data" / "banking77.json")
    test_rows = _read_split(tmp_path / "jsonl" / "test.jsonl")
    assert test_rows
    for row in test_rows:
        assert row.metadata.group_id != row.gold
        assert not str(row.metadata.group_id).startswith("intent:")
        assert len(row.question.criteria) == 77
    manifest = __import__("json").loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["splits"]) >= {"train", "validation", "calibration", "test"}


def test_sst5_score_levels_frozen(tmp_path: Path) -> None:
    convert_sst5(tmp_path, fixture=FIXTURES / "data" / "sst5.json")
    spec = load_criteria(criteria_path("sst5"))
    test_rows = _read_split(tmp_path / "jsonl" / "test.jsonl")
    assert test_rows[0].type == "score"
    assert test_rows[0].question.criteria == spec["criteria"]
    assert 0 <= test_rows[0].gold <= 4


def test_boolq_original_jsonl_uses_title(tmp_path: Path) -> None:
    dev = tmp_path / "dev.jsonl"
    dev.write_text(
        '{"title":"Pluto","passage":"Pluto is a dwarf planet.","question":"Is Pluto a dwarf planet?","answer":true}\n',
        encoding="utf-8",
    )
    from jev.data.boolq import _load_original_boolq_jsonl

    rows = _load_original_boolq_jsonl(dev, "validation")
    assert rows[0]["title"] == "Pluto"


def test_boolq_groups_by_title_and_uses_validation_as_test(tmp_path: Path) -> None:
    convert_boolq(tmp_path, fixture=FIXTURES / "data" / "boolq.json")
    names = ("train", "validation", "calibration", "test")
    splits = {name: _read_split(tmp_path / "jsonl" / f"{name}.jsonl") for name in names}
    assert_no_group_overlap(splits)
    test_titles = {row.metadata.group_id for row in splits["test"]}
    assert "pluto" in test_titles
    assert "hiddenpage" not in test_titles
    for row in splits["test"]:
        assert row.type == "noul"
        assert row.metadata.group_id == row.state["title"].lower()


def test_clinc150_includes_oos_and_151_options(tmp_path: Path) -> None:
    convert_clinc150(tmp_path, fixture=FIXTURES / "data" / "clinc150.json")
    test_rows = _read_split(tmp_path / "jsonl" / "test.jsonl")
    assert any(r.gold == "out_of_scope" for r in test_rows)
    assert len(test_rows[0].question.criteria) == 151
    spec = load_criteria(REPO_ROOT / "assets" / "criteria" / "clinc150.json")
    assert len(spec["criteria"]) == 151


def test_clinc_integer_classlabel_maps_to_intent_names() -> None:
    from jev.data.clinc150 import decode_clinc_intent

    names = ["translate", "oos", "balance"]
    assert decode_clinc_intent(0, names) == "translate"
    assert decode_clinc_intent(1, names) == "out_of_scope"
    assert decode_clinc_intent("oos", names) == "out_of_scope"
    assert decode_clinc_intent("balance", None) == "balance"


def test_wikispeedia_replay_stack_and_cap(tmp_path: Path) -> None:
    pairs = replay_path(["Water", "Quantum_mechanics", "Photon", "<", "Albert_Einstein"])
    assert pairs == [
        ("Water", "Quantum_mechanics"),
        ("Quantum_mechanics", "Photon"),
        ("Quantum_mechanics", "Albert_Einstein"),
    ]
    convert_wikispeedia(tmp_path, fixture=FIXTURES / "data" / "wikispeedia.json")
    all_rows = []
    for name in ("train", "validation", "calibration", "test"):
        all_rows.extend(_read_split(tmp_path / "jsonl" / f"{name}.jsonl"))
    assert all_rows
    hub_rows = [r for r in all_rows if r.state["current"] == "Hub"]
    assert hub_rows
    assert all(len(r.question.criteria) <= 255 for r in hub_rows)
    assert all(r.gold in r.question.criteria for r in hub_rows)
    assert all(r.metadata.group_id == r.state["target"] for r in all_rows)
    menu = sample_menu("Photon", [f"x{i}" for i in range(300)] + ["Photon"], cap=255, seed_key="k")
    assert len(menu) == 255
    assert "Photon" in menu


def test_wikispeedia_snap_parser(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "articles.tsv").write_text("# name\nWater\nPhoton\nAlbert_Einstein\n", encoding="utf-8")
    (snap / "links.tsv").write_text(
        "# src\tdst\nWater\tPhoton\nPhoton\tAlbert_Einstein\nWater\tAlbert_Einstein\n",
        encoding="utf-8",
    )
    (snap / "paths_finished.tsv").write_text(
        "# ip\tts\tdur\tpath\trating\nabc\t0\t1\tWater;Photon;<;Albert_Einstein\t5\n",
        encoding="utf-8",
    )
    from jev.data.wikispeedia import payload_from_snap_dir

    payload = payload_from_snap_dir(snap)
    assert "Water" in payload["articles"]
    assert "Photon" in payload["graph"]["Water"]
    assert payload["paths"][0]["tokens"][-1] == "Albert_Einstein"
    pairs = replay_path(payload["paths"][0]["tokens"])
    assert ("Water", "Photon") in pairs
    assert ("Water", "Albert_Einstein") in pairs
