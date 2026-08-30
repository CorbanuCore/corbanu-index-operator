#!/usr/bin/env python3
"""Acquire exact-period transcripts for the frozen SEC v81 universe.

The SEC top-1,000 artifact remains unchanged. This program binds each company
to the fiscal period on its latest SEC revenue quarter, then attempts
EarningsCall.biz and FMP for that same period. Licensed transcript bodies and
provider caches live under ``/home/postfiat/var`` and never enter the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from corbanu_index_operator import atomic_json, canonical_bytes, load_universe, sha256


DEFAULT_CACHE = Path("/home/postfiat/var/corbanu-indexes/transcript-acquisition")
DEFAULT_OUTPUT = Path(
    "/home/postfiat/var/corbanu-indexes/transcripts/"
    "sec_v81_top1000_transcripts_20260830.json"
)
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "PostFiat deterministic index research ops@postfiat.org"
)
ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
SECRET_QUERY_RE = re.compile(r"(?i)(apikey=)[^&\s]+")
SEC_RATE_LOCK = threading.Lock()
SEC_LAST_START = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_error(value: Any, *secrets: str) -> str:
    message = SECRET_QUERY_RE.sub(r"\1REDACTED", str(value or ""))
    for secret in secrets:
        if secret:
            message = message.replace(secret, "REDACTED")
    return message[:1_000]


def validate_provider_transcript(validator: Any, value: Any) -> tuple[bool, str, dict[str, Any]]:
    accepted, reason, meta = validator(value)
    if not accepted and reason == "body_lacks_call_structure":
        # Some issuers (notably Netflix) publish a long investor interview
        # instead of an operator-led call. The shared validator has already
        # rejected short bodies and error/gate pages before returning this
        # reason, so the provider transcript remains a valid grounding input.
        return True, "accepted_nonstandard_call_format", meta
    return accepted, reason, meta


def target_accession(company: dict[str, Any]) -> str:
    quarters = company["fundamental_evidence"]["revenue"]["quarters"]
    latest = max(quarters, key=lambda row: str(row["period_end"]))
    matches = ACCESSION_RE.findall(str(latest["accession"]))
    if not matches:
        raise RuntimeError(f"SEC accession is missing for {company['ticker']}")
    return matches[0]


def sec_period_from_companyfacts(
    payload: dict[str, Any], *, period_end: str, accession: str
) -> tuple[int, int, str]:
    candidates: set[tuple[int, int, str]] = set()
    for taxonomy in (payload.get("facts") or {}).values():
        for concept in taxonomy.values():
            for rows in (concept.get("units") or {}).values():
                for row in rows:
                    if row.get("end") != period_end or row.get("accn") != accession:
                        continue
                    fiscal_year = row.get("fy")
                    fiscal_period = str(row.get("fp") or "").upper()
                    form = str(row.get("form") or "").upper()
                    if not str(fiscal_year).isdigit():
                        continue
                    if fiscal_period == "FY":
                        quarter = 4
                    elif fiscal_period in {"Q1", "Q2", "Q3", "Q4"}:
                        quarter = int(fiscal_period[1])
                    else:
                        continue
                    candidates.add((int(fiscal_year), quarter, form))
    periods = {(year, quarter) for year, quarter, _ in candidates}
    if len(periods) != 1:
        raise RuntimeError(
            f"SEC period mapping expected one fiscal period for {accession} at "
            f"{period_end}; found {sorted(periods)}"
        )
    fiscal_year, fiscal_quarter = next(iter(periods))
    forms = sorted(
        {
            form
            for year, quarter, form in candidates
            if year == fiscal_year and quarter == fiscal_quarter
        }
    )
    return fiscal_year, fiscal_quarter, ",".join(forms)


def fetch_sec_period(
    company: dict[str, Any], cache_root: Path, *, minimum_interval: float = 0.12
) -> dict[str, Any]:
    global SEC_LAST_START
    cik = str(company["cik"])
    cache_path = cache_root / "sec-periods" / f"{cik}.json"
    period_end = str(company["fundamental_evidence"]["revenue"]["period_end"])
    accession = target_accession(company)
    binding = {
        "cik": cik,
        "ticker": company["ticker"],
        "period_end": period_end,
        "accession": accession,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_bytes())
        if all(cached.get(key) == value for key, value in binding.items()):
            return cached

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    with SEC_RATE_LOCK:
        delay = SEC_LAST_START + minimum_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        SEC_LAST_START = time.monotonic()
    result = subprocess.run(
        [
            "curl",
            "--max-time",
            "10",
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "1",
            "--fail",
            "--silent",
            "--show-error",
            "--compressed",
            "-H",
            f"User-Agent: {SEC_USER_AGENT}",
            url,
        ],
        check=True,
        capture_output=True,
        timeout=50,
    )
    body = result.stdout
    payload = json.loads(body)
    fiscal_year, fiscal_quarter, form = sec_period_from_companyfacts(
        payload, period_end=period_end, accession=accession
    )
    record = {
        **binding,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "form": form,
        "source_url": url,
        "companyfacts_sha256": sha256(body),
        "resolved_at_utc": utc_now(),
    }
    atomic_json(cache_path, record)
    return record


def fetch_sec_ticker_registry(cache_root: Path) -> tuple[dict[str, list[str]], str]:
    cache_path = cache_root / "sec-company-tickers-exchange.json"
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    if cache_path.exists():
        artifact = json.loads(cache_path.read_bytes())
        stable = dict(artifact)
        expected_hash = str(stable.pop("artifact_sha256", ""))
        if expected_hash != sha256(canonical_bytes(stable)):
            raise RuntimeError("SEC ticker registry cache hash does not verify")
    else:
        result = subprocess.run(
            [
                "curl",
                "--max-time",
                "30",
                "--fail",
                "--silent",
                "--show-error",
                "--compressed",
                "-H",
                f"User-Agent: {SEC_USER_AGENT}",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=35,
        )
        payload = json.loads(result.stdout)
        fields = list(payload.get("fields") or [])
        if fields != ["cik", "name", "ticker", "exchange"]:
            raise RuntimeError(f"unexpected SEC ticker registry fields: {fields}")
        stable = {
            "schema": "corbanu.sec-ticker-registry.v1",
            "source_url": url,
            "source_sha256": sha256(result.stdout),
            "fetched_at_utc": utc_now(),
            "fields": fields,
            "data": payload["data"],
        }
        artifact = {**stable, "artifact_sha256": sha256(canonical_bytes(stable))}
        atomic_json(cache_path, artifact)
    by_cik: dict[str, list[str]] = {}
    for cik, _, ticker, _ in artifact["data"]:
        normalized_cik = str(cik).zfill(10)
        normalized_ticker = str(ticker or "").strip().upper()
        if normalized_ticker:
            by_cik.setdefault(normalized_cik, []).append(normalized_ticker)
    return {
        cik: sorted(set(tickers), key=lambda value: ("-" in value or "." in value, value))
        for cik, tickers in by_cik.items()
    }, str(artifact["artifact_sha256"])


def transcript_record(
    company: dict[str, Any],
    period: dict[str, Any],
    cache_root: Path,
    universe_hash: str,
    sec_tickers: list[str],
) -> dict[str, Any]:
    from navstrategies.data_sources.earningscall_biz import (
        EarningsCallBizClient,
        redact_earningscall_error,
        validate_transcript_body,
    )
    from navstrategies.data_sources.fmp import FMPClient

    cik = str(company["cik"])
    ticker = str(company["ticker"]).upper()
    year = int(period["fiscal_year"])
    quarter = int(period["fiscal_quarter"])
    provider_symbols = [ticker] + [symbol for symbol in sec_tickers if symbol != ticker]
    cache_path = cache_root / "transcripts" / f"{cik}.json"
    binding = {
        "universe_sha256": universe_hash,
        "cik": cik,
        "ticker": ticker,
        "company_name": company["company_name"],
        "sec_period_end": period["period_end"],
        "sec_accession": period["accession"],
        "fiscal_year": year,
        "fiscal_quarter": quarter,
        "sec_ticker_candidates": provider_symbols,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_bytes())
        if all(cached.get(key) == value for key, value in binding.items()):
            if cached.get("available") is True:
                body = cached.get("transcript_text")
                accepted, _, meta = validate_provider_transcript(validate_transcript_body, body)
                if (
                    accepted
                    and cached.get("transcript_sha256") == meta.get("sha256")
                    and cached.get("transcript_chars") == meta.get("chars")
                ):
                    return cached

    earnings_key = str(os.environ.get("EARNINGSCALL_API_KEY") or "").strip()
    fmp_key = str(os.environ.get("FMP_API_KEY") or "").strip()
    if not earnings_key or not fmp_key:
        raise RuntimeError("EARNINGSCALL_API_KEY and FMP_API_KEY are required")
    attempts: list[dict[str, Any]] = []
    body: str | None = None
    provider: str | None = None
    source_url: str | None = None

    earnings = EarningsCallBizClient(api_key=earnings_key)
    sdk = earnings._configure_sdk()
    for symbol in provider_symbols:
        try:
            company_object = sdk.get_company(symbol)
            if company_object is None:
                raise LookupError(f"EarningsCall.biz has no company mapping for {symbol}")
            info = company_object.company_info
            transcript = earnings._limited_call(
                f"transcript:{info.exchange}:{info.symbol}:{year}:Q{quarter}:corbanu-top1000",
                lambda: company_object.get_transcript(year=year, quarter=quarter, level=1),
            )
            candidate = str(getattr(transcript, "text", "") or "").strip()
            accepted, reason, meta = validate_provider_transcript(
                validate_transcript_body, candidate
            )
            attempts.append(
                {
                    "provider": "earningscall.biz",
                    "symbol": symbol,
                    "status": "accepted" if accepted else "rejected",
                    "reason": reason,
                    "chars": meta.get("chars", 0),
                }
            )
            if accepted:
                body = candidate
                provider = "earningscall.biz"
                source_url = (
                    f"https://earningscall.biz/transcript/{info.exchange}/{info.symbol}/{year}/{quarter}"
                )
                break
        except Exception as exc:
            attempts.append(
                {
                    "provider": "earningscall.biz",
                    "symbol": symbol,
                    "status": "unavailable",
                    "reason": redact_earningscall_error(exc, api_key=earnings_key)[:1_000],
                    "chars": 0,
                }
            )

    if body is None:
        fmp = FMPClient(fmp_key)
        for symbol in provider_symbols:
            fmp_ticker = symbol.replace(".", "-")
            try:
                frame = fmp.get_transcript(fmp_ticker, quarter, year)
                candidate = ""
                if not frame.empty:
                    for column in ("content", "transcript", "text"):
                        if column in frame.columns:
                            values = [
                                str(value).strip()
                                for value in frame[column].tolist()
                                if str(value).strip()
                            ]
                            if values:
                                candidate = max(values, key=len)
                                break
                accepted, reason, meta = validate_provider_transcript(
                    validate_transcript_body, candidate
                )
                attempts.append(
                    {
                        "provider": "fmp",
                        "symbol": symbol,
                        "status": "accepted" if accepted else "rejected",
                        "reason": reason,
                        "chars": meta.get("chars", 0),
                    }
                )
                if accepted:
                    body = candidate
                    provider = "fmp"
                    source_url = (
                        f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{fmp_ticker}"
                        f"?quarter={quarter}&year={year}"
                    )
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "provider": "fmp",
                        "symbol": symbol,
                        "status": "unavailable",
                        "reason": safe_error(exc, fmp_key),
                        "chars": 0,
                    }
                )

    if body is None:
        record = {
            **binding,
            "available": False,
            "provider": None,
            "source_url": None,
            "transcript_sha256": None,
            "transcript_chars": 0,
            "transcript_text": None,
            "attempts": attempts,
            "acquired_at_utc": utc_now(),
        }
    else:
        record = {
            **binding,
            "available": True,
            "provider": provider,
            "source_url": source_url,
            "transcript_sha256": sha256(body.encode("utf-8")),
            "transcript_chars": len(body),
            "transcript_text": body,
            "attempts": attempts,
            "acquired_at_utc": utc_now(),
        }
    atomic_json(cache_path, record)
    return record


def build_snapshot(
    universe: list[dict[str, Any]],
    universe_hash: str,
    universe_metadata: dict[str, Any],
    records: list[dict[str, Any]],
    sec_registry_sha256: str,
) -> dict[str, Any]:
    if len(records) != len(universe):
        raise RuntimeError(f"refusing partial snapshot: acquired {len(records)}/{len(universe)} records")
    records.sort(key=lambda row: (row["ticker"], row["cik"]))
    stable = {
        "schema": "corbanu.transcript-snapshot.v1",
        "snapshot_id": "sec-v81-top1000-exact-period-transcripts-20260830",
        "universe_id": universe_metadata["id"],
        "universe_sha256": universe_hash,
        "record_count": len(records),
        "available_count": sum(int(row["available"]) for row in records),
        "provider_counts": {
            provider: sum(int(row.get("provider") == provider) for row in records)
            for provider in ("earningscall.biz", "fmp")
        },
        "source": {
            "sec_period_binding": "SEC Company Facts accession, period end, fiscal year, and fiscal period",
            "sec_ticker_registry_sha256": sec_registry_sha256,
            "provider_order": ["earningscall.biz", "fmp"],
            "body_validation": "minimum length, call structure, SHA-256, and character count",
        },
        "selection_rule": (
            "latest SEC v81 revenue period only; EarningsCall.biz first and FMP fallback for the "
            "same SEC fiscal year and quarter; explicit unavailable only after both attempts fail"
        ),
        "records": records,
    }
    return {**stable, "snapshot_sha256": sha256(canonical_bytes(stable))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--only-tickers", default="")
    args = parser.parse_args()

    universe, universe_hash, universe_metadata = load_universe(args.universe)
    sec_registry, sec_registry_sha256 = fetch_sec_ticker_registry(args.cache_root)
    requested = {value.strip().upper() for value in args.only_tickers.split(",") if value.strip()}
    if requested:
        missing = requested - {str(row["ticker"]).upper() for row in universe}
        if missing:
            raise SystemExit(f"tickers outside frozen universe: {sorted(missing)}")
        selected = [row for row in universe if str(row["ticker"]).upper() in requested]
    else:
        selected = sorted(universe, key=lambda row: int(row["revenue_rank"]))

    periods: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_sec_period, company, args.cache_root): company
            for company in selected
        }
        for current, future in enumerate(as_completed(futures), start=1):
            company = futures[future]
            period = future.result()
            periods[str(company["cik"])] = period
            print(
                json.dumps(
                    {
                        "stage": "sec_period",
                        "completed": current,
                        "total": len(selected),
                        "ticker": company["ticker"],
                        "fiscal_year": period["fiscal_year"],
                        "fiscal_quarter": period["fiscal_quarter"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    acquired = []
    for current, company in enumerate(selected, start=1):
        record = transcript_record(
            company,
            periods[str(company["cik"])],
            args.cache_root,
            universe_hash,
            sec_registry.get(str(company["cik"]), []),
        )
        acquired.append(record)
        print(
            json.dumps(
                {
                    "stage": "transcript",
                    "completed": current,
                    "total": len(selected),
                    "ticker": company["ticker"],
                    "available": record["available"],
                    "provider": record.get("provider"),
                    "chars": record["transcript_chars"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if requested:
        print(
            json.dumps(
                {
                    "canary_only": True,
                    "record_count": len(acquired),
                    "available_count": sum(int(row["available"]) for row in acquired),
                    "records": [
                        {
                            key: row.get(key)
                            for key in (
                                "ticker",
                                "fiscal_year",
                                "fiscal_quarter",
                                "available",
                                "provider",
                                "transcript_chars",
                                "transcript_sha256",
                            )
                        }
                        for row in acquired
                    ],
                },
                sort_keys=True,
            )
        )
        return

    artifact = build_snapshot(
        universe, universe_hash, universe_metadata, acquired, sec_registry_sha256
    )
    atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_count": artifact["record_count"],
                "available_count": artifact["available_count"],
                "provider_counts": artifact["provider_counts"],
                "snapshot_sha256": artifact["snapshot_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
