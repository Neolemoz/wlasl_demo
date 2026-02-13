from src.infer import decide_unknown


def test_low_confidence() -> None:
    status, reason = decide_unknown(
        [{"label": "a", "score": 0.5}, {"label": "b", "score": 0.4}],
        T=0.99,
        M=0.1,
    )
    assert status == "unknown"
    assert reason == "low_confidence"


def test_ambiguous() -> None:
    status, reason = decide_unknown(
        [{"label": "a", "score": 0.6}, {"label": "b", "score": 0.55}],
        T=0.5,
        M=0.2,
    )
    assert status == "unknown"
    assert reason == "ambiguous"


def test_ok() -> None:
    status, reason = decide_unknown(
        [{"label": "a", "score": 0.8}, {"label": "b", "score": 0.4}],
        T=0.5,
        M=0.2,
    )
    assert status == "ok"
    assert reason is None
