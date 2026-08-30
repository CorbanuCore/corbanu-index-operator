from decimal import Decimal
import json

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


def test_score_request_binds_transcript_body_and_hash() -> None:
    request = {
        "mandate": {"title": "Grid", "phrase": "Direct grid modernization exposure"},
    }
    rubric = {
        "basket_description": "Direct exposure to grid modernization infrastructure.",
        "scoring_heuristic": "Score direct strategic exposure to the stated theme.",
        "score_0": "No exposure to grid infrastructure or related services.",
        "score_25": "Incidental exposure with no material strategic importance.",
        "score_50": "Material exposure, but not a major strategic business pillar.",
        "score_75": "Direct exposure that is a major strategic business pillar.",
        "score_100": "A defining pure-play exposure to grid modernization infrastructure.",
    }
    company = {"ticker": "GEV", "company_name": "GE Vernova", "cik": "0001996810"}
    body = "Operator: " + "grid modernization and transformers " * 100
    transcript = {
        "available": True,
        "provider": "earningscall.biz",
        "fiscal_year": 2026,
        "fiscal_quarter": 2,
        "source_url": "https://example.invalid/transcript",
        "transcript_sha256": corbanu.sha256(body.encode()),
        "transcript_chars": len(body),
        "transcript_text": body,
    }
    payload = corbanu.score_request(request, rubric, company, transcript)
    prompt = payload["messages"][0]["content"]
    assert corbanu.GENERIC_SCORING_PROMPT_PREFIX in prompt
    assert corbanu.GENERIC_SCORING_PROMPT_SUFFIX in prompt
    input_text = prompt.removeprefix(corbanu.GENERIC_SCORING_PROMPT_PREFIX).removesuffix(
        corbanu.GENERIC_SCORING_PROMPT_SUFFIX
    )
    packet = json.loads(input_text)
    assert packet["company"]["ticker"] == "GEV"
    assert packet["supplemental_transcript_information"]["transcript_text"] == body
    assert packet["supplemental_transcript_information"]["transcript_sha256"] == transcript["transcript_sha256"]


def test_score_request_marks_missing_transcript_explicitly() -> None:
    request = {"mandate": {"title": "Grid", "phrase": "Direct grid exposure"}}
    rubric = {
        "basket_description": "Direct exposure to grid modernization infrastructure.",
        "scoring_heuristic": "Score direct strategic exposure to the stated theme.",
        "score_0": "No exposure to grid infrastructure or related services.",
        "score_25": "Incidental exposure with no material strategic importance.",
        "score_50": "Material exposure, but not a major strategic business pillar.",
        "score_75": "Direct exposure that is a major strategic business pillar.",
        "score_100": "A defining pure-play exposure to grid modernization infrastructure.",
    }
    company = {"ticker": "NONE", "company_name": "No Transcript", "cik": "0000000001"}
    transcript = {
        "available": False,
        "provider": None,
        "fiscal_year": None,
        "fiscal_quarter": None,
        "source_url": None,
        "transcript_sha256": None,
        "transcript_chars": 0,
        "transcript_text": None,
    }
    payload = corbanu.score_request(request, rubric, company, transcript)
    prompt = payload["messages"][0]["content"]
    assert '"available":false' in prompt
    assert '"transcript_text":null' in prompt
