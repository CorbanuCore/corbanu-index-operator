#!/usr/bin/env python3
"""Run a transcript-grounded sample index on both admitted H200 verifiers."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corbanu_index_operator import (
    DEFAULT_TRANSCRIPTS,
    DEFAULT_UNIVERSE,
    MODEL_ID,
    MODEL_PROFILE,
    MODEL_REVISION,
    PRIMARY_ENDPOINT,
    REPLAY_ENDPOINT,
    atomic_json,
    build_weights,
    canonical_bytes,
    complete,
    load_transcripts,
    load_universe,
    rubric_request,
    score_request,
    sha256,
    validate_rubric,
    validate_score,
)


DEFAULT_TICKERS = (
    "AAPL,AMD,AMZN,AVGO,COIN,CRM,INTC,LITE,LMT,META,MRVL,MSFT,MU,NVDA,"
    "ORCL,PLTR,SNDK,TSLA,WMT"
)


def paired_completion(payload: dict) -> tuple:
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_future = pool.submit(complete, PRIMARY_ENDPOINT, payload)
        replay_future = pool.submit(complete, REPLAY_ENDPOINT, payload)
        return primary_future.result(), replay_future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/postfiat/var/corbanu-indexes/samples/generative-ai-transcript-sample-20260830.json"),
    )
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    universe, universe_hash, universe_metadata = load_universe(args.universe)
    transcripts, transcript_hash, transcript_metadata = load_transcripts(args.transcripts, universe)
    requested = [value.strip().upper() for value in args.tickers.split(",") if value.strip()]
    by_ticker = {row["ticker"].upper(): row for row in universe}
    missing = [ticker for ticker in requested if ticker not in by_ticker]
    if missing:
        raise SystemExit(f"sample tickers are outside the frozen universe: {missing}")
    sample = [by_ticker[ticker] for ticker in requested]

    request = {
        "schema": "corbanu.index-request.v1",
        "mandate": {
            "title": "Generative AI & Compute Infrastructure",
            "phrase": (
                "Companies central to developing and deploying generative AI, including foundation-model "
                "developers and the specialized semiconductors, networking, memory, cloud, and data-center "
                "infrastructure required to train and serve those models. Exclude companies that merely use "
                "AI as an incidental feature without direct, strategically important exposure."
            ),
        },
        "classification": {"relevance_cutoff": 70},
        "construction": {"weighting": "fundamental"},
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "execution_profile": MODEL_PROFILE,
        },
    }
    rubric_payload = rubric_request(request)
    primary_rubric, replay_rubric = paired_completion(rubric_payload)
    validate_rubric(primary_rubric.parsed)
    validate_rubric(replay_rubric.parsed)
    if primary_rubric.raw.encode("utf-8") != replay_rubric.raw.encode("utf-8"):
        raise RuntimeError("sample rubric was not byte-identical")

    scores = []
    comparisons = []
    for current, company in enumerate(sample, start=1):
        transcript = transcripts[company["cik"]]
        payload = score_request(request, primary_rubric.parsed, company, transcript)
        primary, replay = paired_completion(payload)
        validate_score(primary.parsed, company["ticker"])
        validate_score(replay.parsed, company["ticker"])
        identical = primary.raw.encode("utf-8") == replay.raw.encode("utf-8")
        if not identical:
            raise RuntimeError(f"sample replay was not byte-identical for {company['ticker']}")
        transcript_public = {
            "available": bool(transcript["available"]),
            "provider": transcript.get("provider"),
            "fiscal_year": transcript.get("fiscal_year"),
            "fiscal_quarter": transcript.get("fiscal_quarter"),
            "sha256": transcript.get("transcript_sha256"),
        }
        scores.append(
            {
                "cik": company["cik"],
                "ticker": company["ticker"],
                **primary.parsed,
                "raw_model_output": primary.raw,
                "transcript": transcript_public,
            }
        )
        comparisons.append(
            {
                "cik": company["cik"],
                "ticker": company["ticker"],
                "request_sha256": primary.request_sha256,
                "primary_content_sha256": primary.content_sha256,
                "replay_content_sha256": replay.content_sha256,
                "byte_identical": identical,
            }
        )
        print(
            json.dumps(
                {
                    "completed": current,
                    "total": len(sample),
                    "ticker": company["ticker"],
                    "score": primary.parsed["score"],
                    "transcript_provider": transcript.get("provider"),
                    "byte_identical": identical,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    holdings = build_weights(request, sample, scores)
    stable = {
        "schema": "corbanu.index-sample.v1",
        "sample_only_not_a_full_index": True,
        "request": request,
        "universe": universe_metadata,
        "universe_sha256": universe_hash,
        "transcript_snapshot": transcript_metadata,
        "transcript_snapshot_sha256": transcript_hash,
        "sample_tickers": requested,
        "rubric": primary_rubric.parsed,
        "rubric_raw_utf8": primary_rubric.raw,
        "rubric_request_sha256": primary_rubric.request_sha256,
        "rubric_content_sha256": primary_rubric.content_sha256,
        "scores": scores,
        "replay_comparisons": comparisons,
        "replay_proof": {
            "comparison_count": len(comparisons),
            "byte_identical_count": sum(int(row["byte_identical"]) for row in comparisons),
            "rubric_byte_identical": True,
            "comparisons_sha256": sha256(canonical_bytes(comparisons)),
        },
        "construction": {
            "relevance_cutoff": 70,
            "formula": "score_multiplier * TTM revenue * exp(0.03 * selected profitability population z-score)",
            "profitability_router": "TTM net income for SEC SIC financials and utilities; TTM free cash flow otherwise",
            "holding_cap": "0.20",
        },
        "constituent_count": len(holdings),
        "holdings": holdings,
    }
    artifact = {**stable, "artifact_sha256": sha256(canonical_bytes(stable))}
    atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "constituent_count": len(holdings),
                "byte_replay": f"{len(comparisons)}/{len(comparisons)}",
                "top_holdings": [
                    {
                        "ticker": row["ticker"],
                        "score": row["score"],
                        "weight_percent": row["weight_percent"],
                    }
                    for row in holdings[:10]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
