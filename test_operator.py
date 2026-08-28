from decimal import Decimal

import corbanu_index_operator as corbanu


def test_canonical_universe_is_exactly_top_1000() -> None:
    rows, universe_hash, metadata = corbanu.load_universe(corbanu.DEFAULT_UNIVERSE)
    assert len(rows) == 1000
    assert len({row["cik"] for row in rows}) == 1000
    assert len(universe_hash) == 64
    assert metadata["company_count"] == 1000
    assert metadata["id"].startswith("sec-registered-us-listed-top1000")
    assert all(Decimal(row["adjusted_scale"]) > 0 for row in rows)


def test_capped_weights_sum_exactly() -> None:
    rows = [
        {"cik": f"{index:010d}", "ticker": f"T{index}", "raw_weight": str(100 - index)}
        for index in range(8)
    ]
    weighted = corbanu.capped_weights(rows, "raw_weight")
    assert sum(row["weight_units"] for row in weighted) == corbanu.WEIGHT_UNITS
    assert max(Decimal(row["weight"]) for row in weighted) <= Decimal("0.20")


def test_validate_request_rejects_public_cap_weighting() -> None:
    request = {
        "request": {
            "schema": "corbanu.index-request.v1",
            "mandate": {"title": "Test index", "phrase": "Direct exposure to test infrastructure"},
            "classification": {"relevance_cutoff": 70},
            "construction": {"weighting": "market_cap"},
            "model": {
                "id": corbanu.MODEL_ID,
                "revision": corbanu.MODEL_REVISION,
                "execution_profile": corbanu.MODEL_PROFILE,
            },
        }
    }
    try:
        corbanu.validate_request(request)
    except ValueError as error:
        assert "cap weighting" in str(error)
    else:
        raise AssertionError("cap weighting unexpectedly accepted")
