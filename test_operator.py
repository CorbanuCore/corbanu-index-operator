from decimal import Decimal
import json
from pathlib import Path

import corbanu_index_operator as corbanu
from build_transcript_snapshot import (
    sec_period_from_companyfacts,
    target_accession,
    validate_provider_transcript,
)


ROOT = Path(__file__).resolve().parent


def admitted_request(*, weighting: str = "fundamental") -> dict:
    return {
        "request": {
            "schema": "corbanu.index-request.v1",
            "mandate": {"title": "Test index", "phrase": "Direct exposure to test infrastructure"},
            "classification": {"relevance_cutoff": 70},
            "construction": {"weighting": weighting},
            "model": {
                "id": corbanu.MODEL_ID,
                "revision": corbanu.MODEL_REVISION,
                "execution_profile": corbanu.MODEL_PROFILE,
                "execution": corbanu.STRICT_EXECUTION_PROFILE.copy(),
            },
        }
    }


def test_h200_bootstrap_contains_complete_strict_fixed32_profile() -> None:
    script = (ROOT / "ops" / "vast_qwen38_fp8_fixed32_strict_onstart.sh").read_text()
    required = (
        "--max-running-requests 32",
        "--attention-backend triton",
        "--enable-deterministic-inference",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--disable-prefill-cuda-graph",
        "--disable-decode-cuda-graph",
        "--random-seed 438916795",
    )
    assert all(flag in script for flag in required)
    assert "--attention-backend fa3" not in script


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
    request = admitted_request(weighting="market_cap")
    try:
        corbanu.validate_request(request)
    except ValueError as error:
        assert "cap weighting" in str(error)
    else:
        raise AssertionError("cap weighting unexpectedly accepted")


def test_validate_request_rejects_partial_deterministic_profile() -> None:
    request = admitted_request()
    request["request"]["model"]["execution"]["overlap_schedule"] = True
    try:
        corbanu.validate_request(request)
    except ValueError as error:
        assert "strict fixed-32 profile" in str(error)
    else:
        raise AssertionError("partial deterministic profile unexpectedly accepted")


def test_public_state_exposes_strict_recovery_progress() -> None:
    store = object.__new__(corbanu.JobStore)
    state = {
        "job_id": "idx_0123456789abcdef",
        "status": "running",
        "stage": "strict_replay_recovery",
        "progress": {"completed": 2002, "total": 2002},
        "recovery": {"completed": 7, "total": 19},
    }
    public = store.public_state(state)
    assert public["recovery"] == {"completed": 7, "total": 19}


def test_registry_and_replay_recipe_use_verified_artifact(tmp_path: Path) -> None:
    job_id = "idx_0123456789abcdef"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    stable_artifact = {
        "schema": "corbanu.index-artifact.v1",
        "valid": True,
        "job_id": job_id,
        "request_sha256": "1" * 64,
        "index_title": "Test index",
        "index_mandate": "Direct exposure to test infrastructure",
        "rubric": {"basket_description": "Test"},
        "rubric_raw_utf8": '{"basket_description":"Test"}',
        "rubric_request_sha256": "2" * 64,
        "rubric_content_sha256": "3" * 64,
        "model": {
            "id": corbanu.MODEL_ID,
            "revision": corbanu.MODEL_REVISION,
            "execution_profile": corbanu.MODEL_PROFILE,
        },
        "universe": {
            "id": "test-universe",
            "as_of_date": "2026-08-30",
            "company_count": 1,
            "universe_sha256": "4" * 64,
        },
        "transcripts": {"snapshot_sha256": "5" * 64, "available_count": 1},
        "construction": {"weighting": "fundamental", "relevance_cutoff": 70},
        "score_vector_sha256": "6" * 64,
        "scores": [
            {
                "cik": "0000000001",
                "ticker": "TEST",
                "score": 75,
                "transcript": {"available": True, "sha256": "7" * 64},
            }
        ],
        "replay_proof": {
            "comparison_count": 1,
            "byte_identical_count": 1,
            "request_identical_count": 1,
            "batch_byte_identical_count": 1,
            "strict_retry_count": 0,
            "comparisons_sha256": "8" * 64,
        },
        "replay_comparisons": [
            {
                "cik": "0000000001",
                "ticker": "TEST",
                "request_sha256": "9" * 64,
                "primary_content_sha256": "a" * 64,
                "replay_content_sha256": "a" * 64,
                "byte_identical": True,
                "batch_byte_identical": True,
                "strict_retry": False,
            }
        ],
        "constituent_count": 1,
        "holdings": [{"ticker": "TEST", "weight_units": corbanu.WEIGHT_UNITS}],
    }
    artifact_hash = corbanu.sha256(corbanu.canonical_bytes(stable_artifact))
    artifact = {**stable_artifact, "artifact_sha256": artifact_hash}
    state = {
        "job_id": job_id,
        "status": "valid",
        "created_at_unix": 1,
        "updated_at_unix": 2,
        "artifact_sha256": artifact_hash,
        "request": admitted_request()["request"],
        "transcripts": stable_artifact["transcripts"],
    }
    (job_dir / "artifact.json").write_bytes(corbanu.canonical_bytes(artifact) + b"\n")
    (job_dir / "job.json").write_bytes(corbanu.canonical_bytes(state) + b"\n")

    store = object.__new__(corbanu.JobStore)
    store.jobs_dir = tmp_path
    store.universe = [{"cik": "0000000001"}]
    store.universe_sha256 = "4" * 64
    store.transcript_snapshot_sha256 = "5" * 64
    registry = store.list_valid_indexes()
    assert registry["index_count"] == 1
    assert registry["indexes"][0]["current_admitted_profile"] is True
    assert registry["indexes"][0]["replay_recipe_path"].endswith("/recipe")

    recipe = store.replay_recipe(job_id)
    assert recipe is not None
    assert recipe["required_inputs"]["transcript_snapshot"]["snapshot_sha256"] == "5" * 64
    assert recipe["scoring_contract"]["company_evidence"][0]["ticker"] == "TEST"
    assert recipe["scoring_contract"]["company_evidence"][0]["scoring_request_sha256"] == "9" * 64
    stable_recipe = dict(recipe)
    recipe_hash = stable_recipe.pop("recipe_sha256")
    assert corbanu.sha256(corbanu.canonical_bytes(stable_recipe)) == recipe_hash

    legacy_stable = dict(stable_artifact)
    legacy_stable["model"] = {
        **stable_artifact["model"],
        "execution_profile": "superseded-profile",
    }
    legacy_hash = corbanu.sha256(corbanu.canonical_bytes(legacy_stable))
    legacy_artifact = {**legacy_stable, "artifact_sha256": legacy_hash}
    state["artifact_sha256"] = legacy_hash
    (job_dir / "artifact.json").write_bytes(
        corbanu.canonical_bytes(legacy_artifact) + b"\n"
    )
    (job_dir / "job.json").write_bytes(corbanu.canonical_bytes(state) + b"\n")
    assert store.list_valid_indexes()["index_count"] == 0
    assert store.replay_recipe(job_id) is None


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


def test_sec_period_mapping_converts_fy_to_q4() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {
                                "end": "2026-05-31",
                                "accn": "0001193125-26-277521",
                                "fy": 2026,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }
    assert sec_period_from_companyfacts(
        payload,
        period_end="2026-05-31",
        accession="0001193125-26-277521",
    ) == (2026, 4, "10-K")


def test_target_accession_uses_annual_source_of_derived_q4() -> None:
    company = {
        "ticker": "ORCL",
        "fundamental_evidence": {
            "revenue": {
                "quarters": [
                    {
                        "period_end": "2026-05-31",
                        "accession": (
                            "0001193125-26-277521;minus:0001193125-26-101045"
                        ),
                    }
                ]
            }
        },
    }
    assert target_accession(company) == "0001193125-26-277521"


def test_nonstandard_long_provider_transcript_is_not_a_false_negative() -> None:
    def validator(_value):
        return False, "body_lacks_call_structure", {
            "chars": 35_709,
            "nonspace_chars": 30_000,
            "sha256": "a" * 64,
        }

    accepted, reason, metadata = validate_provider_transcript(
        validator, "long investor interview"
    )
    assert accepted is True
    assert reason == "accepted_nonstandard_call_format"
    assert metadata["chars"] == 35_709
