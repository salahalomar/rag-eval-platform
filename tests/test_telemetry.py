import time

import pytest

from rag.telemetry import StageTimer


def test_records_a_stage() -> None:
    timer = StageTimer()
    with timer.stage("dense"):
        time.sleep(0.01)
    assert "dense" in timer.timings_ms
    assert timer.timings_ms["dense"] >= 10.0


def test_repeated_stages_accumulate_rather_than_overwrite() -> None:
    # A batched stage must report its total, not its last iteration.
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("embed"):
            time.sleep(0.005)
    assert len(timer.timings_ms) == 1
    assert timer.timings_ms["embed"] >= 15.0


def test_duration_is_recorded_even_when_the_stage_raises() -> None:
    timer = StageTimer()
    with pytest.raises(RuntimeError), timer.stage("rerank"):
        time.sleep(0.005)
        raise RuntimeError("model blew up")
    assert timer.timings_ms["rerank"] >= 5.0


def test_record_accepts_externally_measured_durations() -> None:
    timer = StageTimer()
    timer.record("generation", 120.5)
    timer.record("generation", 9.5)
    assert timer.timings_ms["generation"] == pytest.approx(130.0)


def test_total_is_the_sum_of_stages() -> None:
    timer = StageTimer()
    timer.record("dense", 40.0)
    timer.record("lexical", 12.0)
    assert timer.total_ms() == pytest.approx(52.0)


def test_as_dict_is_rounded_and_detached() -> None:
    timer = StageTimer()
    timer.record("dense", 1.23456)
    snapshot = timer.as_dict()
    assert snapshot == {"dense": 1.235}
    timer.record("dense", 1.0)
    assert snapshot == {"dense": 1.235}  # snapshot must not track later mutation


def test_stage_order_is_preserved() -> None:
    timer = StageTimer()
    for name in ("dense", "lexical", "fusion", "rerank"):
        timer.record(name, 1.0)
    assert list(timer.timings_ms) == ["dense", "lexical", "fusion", "rerank"]
