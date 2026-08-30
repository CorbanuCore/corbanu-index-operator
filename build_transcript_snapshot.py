#!/usr/bin/env python3
"""Freeze the accepted NavStrategies transcript store for the SEC v81 universe.

Transcript bodies are licensed research inputs and are written under
``/home/postfiat/var`` by default, never into the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from corbanu_index_operator import canonical_bytes, load_universe, sha256


SOURCE_TITLE_PERIOD = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)
PROVIDER_PRIORITY = {"earningscall": 0, "earningscall.biz": 0, "fmp": 1}


def parse_period(row: dict[str, Any]) -> tuple[int, int] | None:
    match = SOURCE_TITLE_PERIOD.search(str(row.get("source_title") or ""))
    if not match:
        return None
    return int(match.group(2)), int(match.group(1))


def normalized_provider(value: Any) -> str | None:
    provider = str(value or "").strip().lower()
    if provider in {"earningscall", "earningscall.biz"}:
        return "earningscall.biz"
    if provider == "fmp":
        return "fmp"
    return None


def select_latest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        provider = normalized_provider(row.get("provider"))
        period = parse_period(row)
        body = row.get("transcript_text")
        if provider is None or period is None or not isinstance(body, str):
            continue
        body_hash = sha256(body.encode("utf-8"))
        if body_hash != str(row.get("transcript_sha256") or ""):
            continue
        if len(body) != int(row.get("transcript_chars") or -1) or len(body) < 1_000:
            continue
        candidates.append((period, -PROVIDER_PRIORITY[provider], provider, row))
    if not candidates:
        return None
    # Newest fiscal period first; EarningsCall.biz wins only within that period.
    _, _, provider, selected = max(candidates, key=lambda item: (item[0], item[1]))
    year, quarter = parse_period(selected) or (None, None)
    body = selected["transcript_text"]
    return {
        "available": True,
        "provider": provider,
        "fiscal_year": year,
        "fiscal_quarter": quarter,
        "source_url": selected.get("source_url"),
        "transcript_sha256": sha256(body.encode("utf-8")),
        "transcript_chars": len(body),
        "transcript_text": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database_url = str(os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required")
    universe, universe_hash, universe_metadata = load_universe(args.universe)
    tickers = sorted({str(row["ticker"]).upper() for row in universe})
    engine = create_engine(database_url)
    with engine.connect() as connection:
        database_rows = [
            dict(row)
            for row in connection.execute(
                text(
                    "SELECT ticker, provider, transcript_date, source_url, source_title, "
                    "transcript_text, transcript_sha256, transcript_chars "
                    "FROM sec__earnings_transcript_bodies WHERE upper(ticker) = ANY(:tickers)"
                ),
                {"tickers": tickers},
            ).mappings()
        ]
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in database_rows:
        by_ticker.setdefault(str(row.get("ticker") or "").upper(), []).append(row)
    records = []
    for company in universe:
        selected = select_latest(by_ticker.get(str(company["ticker"]).upper(), []))
        if selected is None:
            selected = {
                "available": False,
                "provider": None,
                "fiscal_year": None,
                "fiscal_quarter": None,
                "source_url": None,
                "transcript_sha256": None,
                "transcript_chars": 0,
                "transcript_text": None,
            }
        records.append(
            {
                "cik": company["cik"],
                "ticker": company["ticker"],
                "company_name": company["company_name"],
                **selected,
            }
        )
    records.sort(key=lambda row: (row["ticker"], row["cik"]))
    stable = {
        "schema": "corbanu.transcript-snapshot.v1",
        "snapshot_id": "sec-v81-top1000-navstrategies-accepted-transcripts-20260830",
        "universe_id": universe_metadata["id"],
        "universe_sha256": universe_hash,
        "record_count": len(records),
        "available_count": sum(int(row["available"]) for row in records),
        "source": {
            "store": "navstrategies.sec__earnings_transcript_bodies",
            "body_validation": "stored SHA-256 and character count must match",
        },
        "selection_rule": (
            "newest parsed fiscal year and quarter; earningscall.biz before FMP "
            "within the same fiscal period; explicit unavailable record otherwise"
        ),
        "records": records,
    }
    artifact = {**stable, "snapshot_sha256": sha256(canonical_bytes(stable))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_bytes(artifact) + b"\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_count": len(records),
                "available_count": artifact["available_count"],
                "snapshot_sha256": artifact["snapshot_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
