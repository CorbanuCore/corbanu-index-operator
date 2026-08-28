#!/usr/bin/env python3
"""Freeze the canonical Corbanu top-1,000 SEC fundamental universe.

This consumes the read-only, SEC-only warehouse produced by
``sec-fundamentals-proxy``.  It never queries a market-data vendor and it does
not infer missing fundamentals.  The latest exact ART observations are the
pipeline's explicit trailing-four-quarter constructions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_WAREHOUSE = Path(
    "/home/postfiat/repos/sec-fundamentals-proxy/data/warehouse/"
    "companyfacts-only-compact-full-20260827-v1.sqlite"
)
DEFAULT_OUTPUT = ROOT / "data" / "sec_top1000_universe_20260827.json"
AS_OF_DATE = dt.date(2026, 8, 27)
MAX_FUNDAMENTAL_AGE_DAYS = 550
ELIGIBLE_EXCHANGES = {"Nasdaq", "NYSE", "NYSE American"}
EXCLUDED_SIC = {"6722", "6726", "6770"}
EXCLUDED_NAME = re.compile(
    r"\b(?:ETF|ETN|ACQUISITION|SPAC|PORTFOLIO|INDEX FUND|MONEY MARKET|WARRANTS?)\b",
    re.IGNORECASE,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def is_financial_or_utility(sic: str) -> bool:
    code = int(sic or "0")
    return 4900 <= code <= 4999 or 6000 <= code <= 6799


def build(warehouse: Path, output: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{warehouse.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    snapshot_id = int(
        connection.execute("SELECT snapshot_id FROM active_snapshot WHERE singleton = 1").fetchone()[0]
    )
    identities = {
        row["cik"]: dict(row)
        for row in connection.execute(
            """
            SELECT i.cik,
                   i.canonical_ticker AS ticker,
                   i.legal_name AS company_name,
                   json_extract(i.identity_json, '$.sic') AS sic,
                   json_extract(i.identity_json, '$.sic_description') AS sic_description,
                   json_extract(i.identity_json, '$.exchanges') AS exchanges
            FROM snapshot_issuers AS snapshot
            JOIN issuers AS i ON i.cik = snapshot.cik
            WHERE snapshot.snapshot_id = ?
            """,
            (snapshot_id,),
        )
    }
    latest: dict[tuple[str, str], tuple[tuple[str, str, str], dict[str, Any]]] = {}
    for raw in connection.execute(
        """
        SELECT fact.cik, fact.ticker, fact.field, fact.value_text,
               fact.period_start, fact.period_end, fact.filed, fact.accession,
               fact.derivation, fact.status, source.source_url, source.source_sha256
        FROM snapshot_issuers AS snapshot
        JOIN facts AS fact ON fact.run_id = snapshot.run_id AND fact.cik = snapshot.cik
        JOIN fact_sources AS source ON source.id = fact.source_id
        WHERE snapshot.snapshot_id = ?
          AND fact.dimension = 'ART'
          AND fact.field IN ('revenue', 'fcf', 'netinc')
          AND fact.unit = 'USD'
        """,
        (snapshot_id,),
    ):
        row = dict(raw)
        key = (row["cik"], row["field"])
        rank = (row["period_end"], row["filed"], row["accession"])
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, row)
    connection.close()

    stale_before = AS_OF_DATE - dt.timedelta(days=MAX_FUNDAMENTAL_AGE_DAYS)
    candidates: list[dict[str, Any]] = []
    excluded_counts: dict[str, int] = {}

    def exclude(reason: str) -> None:
        excluded_counts[reason] = excluded_counts.get(reason, 0) + 1

    for cik, identity in identities.items():
        exchanges = {
            str(exchange)
            for exchange in json.loads(identity.get("exchanges") or "[]")
            if exchange
        }
        if not exchanges.intersection(ELIGIBLE_EXCHANGES):
            exclude("ineligible_exchange")
            continue
        name = str(identity.get("company_name") or "")
        sic = str(identity.get("sic") or "")
        if sic in EXCLUDED_SIC or EXCLUDED_NAME.search(name):
            exclude("fund_spac_or_non_operating_vehicle")
            continue
        observations = {
            field: latest.get((cik, field), (None, None))[1]
            for field in ("revenue", "fcf", "netinc")
        }
        revenue = observations["revenue"]
        if revenue is None:
            exclude("missing_ttm_revenue")
            continue
        revenue_value = Decimal(revenue["value_text"])
        if revenue_value <= 0:
            exclude("nonpositive_ttm_revenue")
            continue
        if dt.date.fromisoformat(revenue["period_end"]) < stale_before:
            exclude("stale_ttm_revenue")
            continue
        accounting_regime = "financial_or_utility" if is_financial_or_utility(sic) else "ordinary_operating"
        profitability_field = "netinc" if accounting_regime == "financial_or_utility" else "fcf"
        profitability = observations[profitability_field]
        if profitability is None:
            exclude("missing_selected_profitability")
            continue
        candidates.append(
            {
                "cik": str(cik).zfill(10),
                "ticker": str(identity["ticker"]).upper(),
                "company_name": name,
                "sec_sic": sic,
                "sec_sic_description": str(identity.get("sic_description") or ""),
                "exchanges": sorted(exchanges),
                "accounting_regime": accounting_regime,
                "profitability_factor": (
                    "trailing_four_quarter_net_income"
                    if profitability_field == "netinc"
                    else "trailing_four_quarter_free_cash_flow"
                ),
                "ttm_revenue": decimal_text(revenue_value),
                "selected_profitability": decimal_text(Decimal(profitability["value_text"])),
                "fundamental_evidence": {
                    "ttm_revenue": revenue,
                    "selected_profitability": profitability,
                },
            }
        )

    candidates.sort(key=lambda row: (-Decimal(row["ttm_revenue"]), row["ticker"], row["cik"]))
    selected = candidates[:1000]
    if len(selected) != 1000:
        raise RuntimeError(f"expected 1,000 eligible issuers, found {len(selected)}")
    if len({row["cik"] for row in selected}) != 1000:
        raise RuntimeError("selected universe contains duplicate CIKs")

    with localcontext() as context:
        context.prec = 60
        profits = [Decimal(row["selected_profitability"]) for row in selected]
        mean = sum(profits) / Decimal(len(profits))
        variance = sum((value - mean) ** 2 for value in profits) / Decimal(len(profits))
        standard_deviation = context.sqrt(variance)
        if standard_deviation == 0:
            raise RuntimeError("selected profitability standard deviation is zero")
        for rank, row in enumerate(selected, start=1):
            z_score = (Decimal(row["selected_profitability"]) - mean) / standard_deviation
            multiplier = context.exp(Decimal("0.03") * z_score)
            adjusted_scale = Decimal(row["ttm_revenue"]) * multiplier
            row["revenue_rank"] = rank
            row["profitability_z_score"] = decimal_text(z_score)
            row["profitability_multiplier"] = decimal_text(multiplier)
            row["adjusted_scale"] = decimal_text(adjusted_scale)

    artifact = {
        "schema": "corbanu.sec-fundamental-universe.v1",
        "universe_id": "sec-us-listed-top1000-ttm-revenue-20260827",
        "as_of_date": AS_OF_DATE.isoformat(),
        "company_count": len(selected),
        "source": {
            "provider": "U.S. Securities and Exchange Commission",
            "warehouse_snapshot_id": snapshot_id,
            "companyfacts_archive_sha256": (
                selected[0]["fundamental_evidence"]["ttm_revenue"]["source_sha256"]
            ),
            "company_tickers_exchange_source": "https://www.sec.gov/files/company_tickers_exchange.json",
            "sec_only": True,
        },
        "methodology": {
            "selection": "top 1,000 eligible U.S.-listed SEC registrants by positive latest TTM revenue",
            "ttm_definition": "latest ART value explicitly constructed from four exact fiscal quarters",
            "freshness_max_days": MAX_FUNDAMENTAL_AGE_DAYS,
            "eligible_exchanges": sorted(ELIGIBLE_EXCHANGES),
            "excluded_sic": sorted(EXCLUDED_SIC),
            "profitability_router": (
                "SEC SIC 4900-4999 and 6000-6799 use TTM net income; all other issuers use TTM free cash flow"
            ),
            "profitability_overlay": "TTM revenue multiplied by exp(0.03 * population z-score of selected profitability)",
            "candidate_count_before_top1000": len(candidates),
            "excluded_counts": excluded_counts,
            "population_profitability_mean": decimal_text(mean),
            "population_profitability_standard_deviation": decimal_text(standard_deviation),
        },
        "companies": selected,
    }
    artifact["universe_sha256"] = sha256(canonical_bytes(artifact))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(artifact) + b"\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = build(args.warehouse, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "company_count": artifact["company_count"],
                "universe_sha256": artifact["universe_sha256"],
                "largest": [row["ticker"] for row in artifact["companies"][:10]],
                "revenue_rank_1000": artifact["companies"][-1]["ticker"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
