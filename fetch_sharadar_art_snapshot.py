#!/usr/bin/env python3
"""Freeze current Sharadar ART fundamentals for the SEC-listed universe."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

from build_sec_top1000_universe import (
    DEFAULT_WAREHOUSE,
    ELIGIBLE_EXCHANGES,
    EXCLUDED_NAME,
    EXCLUDED_SIC,
    canonical_bytes,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS = Path("/home/postfiat/.config/navstrategies/credentials.env")
DEFAULT_OUTPUT = ROOT / "data" / "sharadar_art_snapshot_20260828.json"
API_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.json"


def api_key(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(("NASDAQ_DATA_LINK_API_KEY=", "QUANDL_API_KEY=", "SHARADAR_API_KEY=")):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise RuntimeError("Sharadar API key is not configured")


def eligible_tickers(warehouse: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{warehouse.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    snapshot_id = int(connection.execute("SELECT snapshot_id FROM active_snapshot WHERE singleton=1").fetchone()[0])
    rows = connection.execute(
        """
        SELECT i.canonical_ticker AS ticker, i.legal_name AS company_name,
               json_extract(i.identity_json, '$.sic') AS sic,
               json_extract(i.identity_json, '$.exchanges') AS exchanges,
               json_extract(i.identity_json, '$.tickers') AS tickers
        FROM snapshot_issuers AS snapshot
        JOIN issuers AS i ON i.cik = snapshot.cik
        WHERE snapshot.snapshot_id = ?
        """,
        (snapshot_id,),
    )
    tickers = []
    for row in rows:
        exchanges = {str(value) for value in json.loads(row["exchanges"] or "[]") if value}
        if not exchanges.intersection(ELIGIBLE_EXCHANGES):
            continue
        if str(row["sic"] or "") in EXCLUDED_SIC or EXCLUDED_NAME.search(str(row["company_name"] or "")):
            continue
        aliases = json.loads(row["tickers"] or "[]")
        tickers.extend(str(value).upper() for value in aliases if value)
        tickers.append(str(row["ticker"]).upper())
    connection.close()
    return sorted(set(tickers))


def fetch(tickers: list[str], key: str, chunk_size: int = 75) -> list[dict[str, Any]]:
    latest: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {}
    for offset in range(0, len(tickers), chunk_size):
        chunk = tickers[offset : offset + chunk_size]
        query = urllib.parse.urlencode(
            {
                "ticker": ",".join(chunk),
                "dimension": "ART",
                "calendardate.gte": "2025-01-01",
                "api_key": key,
            }
        )
        url = f"{API_URL}?{query}"
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)["datatable"]
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 7:
                    raise
                retry_after = int(error.headers.get("Retry-After", "10"))
                time.sleep(max(5, min(retry_after, 60)))
        time.sleep(1)
        columns = [column["name"] for column in payload["columns"]]
        for values in payload["data"]:
            source = dict(zip(columns, values))
            ticker = str(source.get("ticker") or "").upper()
            row = {
                "ticker": ticker,
                "dimension": source.get("dimension"),
                "calendardate": source.get("calendardate"),
                "reportperiod": source.get("reportperiod"),
                "datekey": source.get("datekey"),
                "lastupdated": source.get("lastupdated"),
                "revenueusd": source.get("revenueusd"),
                "fcf": source.get("fcf"),
                "netinc": source.get("netinc"),
                "fxusd": source.get("fxusd"),
            }
            rank = (str(row["calendardate"] or ""), str(row["datekey"] or ""))
            if ticker and (ticker not in latest or rank > latest[ticker][0]):
                latest[ticker] = (rank, row)
    return [latest[ticker][1] for ticker in sorted(latest)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tickers = eligible_tickers(args.warehouse)
    rows = fetch(tickers, api_key(args.credentials))
    artifact = {
        "schema": "corbanu.sharadar-art-snapshot.v1",
        "as_of_date": "2026-08-28",
        "requested_ticker_count": len(tickers),
        "returned_ticker_count": len(rows),
        "query": {
            "table": "SHARADAR/SF1",
            "dimension": "ART",
            "calendardate_gte": "2025-01-01",
            "selection": "latest calendardate then latest datekey per ticker",
        },
        "rows": rows,
    }
    artifact["snapshot_sha256"] = hashlib.sha256(canonical_bytes(artifact)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(artifact) + b"\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "snapshot_sha256": artifact["snapshot_sha256"]}))


if __name__ == "__main__":
    main()
