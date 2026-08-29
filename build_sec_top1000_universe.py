#!/usr/bin/env python3
"""Build the canonical Corbanu top-1,000 universe from frozen SEC artifacts.

Membership and trailing-four-quarter revenue come from the validated v77 SEC
cohort. Profitability uses the verified v80 capex-policy shards and v81
net-income-taxonomy overlay, plus the reviewed v78 filing-only capex repairs.
No vendor value, ticker map, fallback, or comparison record is consumed here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = Path("/mnt/HC_Volume_106212907/sec-fundamentals-proxy/runs")
DEFAULT_BENCHMARK = RUNS_ROOT / "sec-top1000-v77-frozen-revenue-benchmark-20260828"
DEFAULT_OUTPUT = ROOT / "data" / "sec_v81_top1000_universe_20260828.json"
V77_FACT_RUNS = tuple(
    RUNS_ROOT
    / "sec-top1000-frozen-minus4-v77-runs-20260828"
    / f"shard-{number:04d}-of-0008"
    for number in range(1, 9)
) + (
    RUNS_ROOT / "sec-top1000-revenue-remaining4-v77-filing-enrichment-20260828",
)
V78_FACT_RUNS = (
    RUNS_ROOT / "sec-top1000-profitability-filing4-v78d-20260828",
)
V80_FACT_RUNS = tuple(
    RUNS_ROOT / f"sec-top1000-profitability-v80-capex-policy-shard{number}-20260828"
    for number in range(1, 9)
)
V81_FACT_RUNS = (RUNS_ROOT / "sec-top1000-netinc-taxonomy-v81-20260828",)
FACT_RUNS = (*V77_FACT_RUNS, *V78_FACT_RUNS, *V80_FACT_RUNS, *V81_FACT_RUNS)
EXPECTED_COHORT_SIZE = 1000
EXPECTED_CUTOFF = "2026-08-27"
MEMBERSHIP_METHODOLOGY = "sec-fundamentals-proxy-v77"
PROFITABILITY_METHODOLOGY = "sec-fundamentals-proxy-v81"
RUN_METHODOLOGY = {
    **{path: MEMBERSHIP_METHODOLOGY for path in V77_FACT_RUNS},
    **{path: "sec-fundamentals-proxy-v78" for path in V78_FACT_RUNS},
    **{path: "sec-fundamentals-proxy-v80" for path in V80_FACT_RUNS},
    **{path: PROFITABILITY_METHODOLOGY for path in V81_FACT_RUNS},
}
RUN_PRIORITY = {
    **{path: 0 for path in V77_FACT_RUNS},
    **{path: 1 for path in V78_FACT_RUNS},
    **{path: 2 for path in V80_FACT_RUNS},
    **{path: 3 for path in V81_FACT_RUNS},
}
COMPANYFACTS_ARCHIVE_SHA256 = "83b34e5cca7fb96746d698ebc99d1560e689375460fbc4d3426cd8adbc142cc1"
SUBMISSIONS_ARCHIVE_SHA256 = "e5cc4db5c85fc701308d7bc6f472c33e2a820218f91bc8acf50537444b25c9e7"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def is_financial_or_utility(sic: str) -> bool:
    code = int(sic or "0")
    return 4900 <= code <= 4999 or 6000 <= code <= 6799


def iter_gzip_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a compact JSON array without materializing a multi-GB fact file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    ended = False
    with gzip.open(path, "rt", encoding="utf-8") as source:
        while not ended:
            chunk = source.read(1024 * 1024)
            if chunk:
                buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise RuntimeError(f"expected JSON array in {path}")
                    started = True
                    buffer = buffer[1:]
                    continue
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    ended = True
                    buffer = buffer[1:]
                    break
                if not buffer:
                    break
                try:
                    value, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object fact in {path}")
                yield value
                buffer = buffer[offset:]
            if not chunk:
                break
    if not started or not ended or buffer.strip():
        raise RuntimeError(f"incomplete JSON array in {path}")


def verify_benchmark(benchmark: Path) -> tuple[dict[str, Any], str, str]:
    manifest_path = benchmark / "manifest.json"
    verification_path = benchmark / "verification.json"
    manifest = json.loads(manifest_path.read_bytes())
    verification = json.loads(verification_path.read_bytes())
    manifest_sha256 = file_sha256(manifest_path)
    if verification.get("valid") is not True or verification.get("failures") != []:
        raise RuntimeError("v77 benchmark verification is not valid")
    if verification.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("v77 benchmark manifest hash does not match verification")
    contract = manifest.get("contract") or {}
    if contract.get("cohort_size") != EXPECTED_COHORT_SIZE:
        raise RuntimeError("v77 benchmark does not contain exactly 1,000 companies")
    if contract.get("cutoff") != EXPECTED_CUTOFF:
        raise RuntimeError("v77 benchmark cutoff changed")
    if contract.get("sharadar_used_for_selection") is not False:
        raise RuntimeError("v77 cohort was not selected exclusively from SEC data")
    for artifact in manifest.get("artifacts") or []:
        path = benchmark / str(artifact["path"])
        if path.stat().st_size != int(artifact["bytes"]):
            raise RuntimeError(f"benchmark artifact size mismatch: {path}")
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"benchmark artifact hash mismatch: {path}")
    cohort_path = benchmark / "cohort.json"
    cohort_sha256 = file_sha256(cohort_path)
    return json.loads(cohort_path.read_bytes()), manifest_sha256, cohort_sha256


def verify_fact_run(path: Path) -> tuple[Path, str]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("sec_only") is not True or manifest.get("vendor_mode") != "sec_only":
        raise RuntimeError(f"fact run is not SEC-only: {path}")
    expected_methodology = RUN_METHODOLOGY.get(path)
    if expected_methodology is None:
        raise RuntimeError(f"unregistered fact run: {path}")
    if manifest.get("methodology_version") != expected_methodology:
        raise RuntimeError(f"unexpected methodology version: {path}")
    if manifest.get("as_of") != f"{EXPECTED_CUTOFF}T23:59:59Z":
        raise RuntimeError(f"unexpected fact-run cutoff: {path}")
    if (manifest.get("companyfacts_source") or {}).get("archive_sha256") != COMPANYFACTS_ARCHIVE_SHA256:
        raise RuntimeError(f"unexpected Company Facts archive: {path}")
    if (manifest.get("submissions_source") or {}).get("archive_sha256") != SUBMISSIONS_ARCHIVE_SHA256:
        raise RuntimeError(f"unexpected submissions archive: {path}")
    facts_artifact = next(
        (item for item in manifest.get("artifacts") or [] if item.get("path") == "facts.json.gz"),
        None,
    )
    if facts_artifact is None:
        raise RuntimeError(f"facts artifact is absent: {path}")
    facts_path = path / "facts.json.gz"
    if facts_path.stat().st_size != int(facts_artifact["bytes"]):
        raise RuntimeError(f"facts artifact size mismatch: {facts_path}")
    if file_sha256(facts_path) != facts_artifact["sha256"]:
        raise RuntimeError(f"facts artifact hash mismatch: {facts_path}")
    return facts_path, file_sha256(manifest_path)


def load_latest_profitability(
    fact_runs: tuple[Path, ...], cohort_ciks: set[str]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    latest: dict[tuple[str, str], tuple[tuple[str, int, str, str, str], dict[str, Any]]] = {}
    by_period: dict[tuple[str, str, str], tuple[tuple[int, str, str, str], dict[str, Any]]] = {}
    manifests: list[dict[str, str]] = []
    for run in fact_runs:
        facts_path, manifest_sha256 = verify_fact_run(run)
        manifests.append(
            {
                "path": str(run),
                "manifest_sha256": manifest_sha256,
                "methodology_version": RUN_METHODOLOGY[run],
            }
        )
        for row in iter_gzip_json_array(facts_path):
            cik = str(row.get("cik") or "").zfill(10)
            field = str(row.get("field") or "")
            if cik not in cohort_ciks or field not in {"fcf", "netinc", "ncfo", "capex"}:
                continue
            if row.get("dimension") != "ART" or row.get("unit") != "USD":
                continue
            if str(row.get("filed") or "") > EXPECTED_CUTOFF:
                continue
            key = (cik, field)
            rank = (
                str(row.get("period_end") or ""),
                RUN_PRIORITY[run],
                str(row.get("filed") or ""),
                str(row.get("accession") or ""),
                str(row.get("source_concept") or ""),
            )
            if key not in latest or rank > latest[key][0]:
                latest[key] = (rank, row)
            period_key = (cik, field, str(row.get("period_end") or ""))
            period_rank = (
                RUN_PRIORITY[run],
                str(row.get("filed") or ""),
                str(row.get("accession") or ""),
                str(row.get("source_concept") or ""),
            )
            if period_key not in by_period or period_rank > by_period[period_key][0]:
                by_period[period_key] = (period_rank, row)

    output = {key: value[1] for key, value in latest.items()}
    for cik in cohort_ciks:
        if (cik, "fcf") in output:
            continue
        periods = sorted(
            {
                period_end
                for candidate_cik, field, period_end in by_period
                if candidate_cik == cik and field == "ncfo"
            },
            reverse=True,
        )
        for period_end in periods:
            ncfo_pair = by_period.get((cik, "ncfo", period_end))
            capex_pair = by_period.get((cik, "capex", period_end))
            if ncfo_pair is None or capex_pair is None:
                continue
            ncfo = ncfo_pair[1]
            capex = capex_pair[1]
            if ncfo.get("period_start") != capex.get("period_start"):
                continue
            fcf_value = Decimal(str(ncfo["value_text"])) + Decimal(str(capex["value_text"]))
            output[(cik, "fcf")] = {
                "cik": cik,
                "ticker": str(ncfo.get("ticker") or capex.get("ticker") or ""),
                "field": "fcf",
                "value_text": decimal_text(fcf_value),
                "period_start": str(ncfo["period_start"]),
                "period_end": period_end,
                "filed": max(str(ncfo["filed"]), str(capex["filed"])),
                "accession": f"{ncfo['accession']};{capex['accession']}",
                "derivation": "fcf=ncfo+capex",
                "status": "exact_derived",
                "source_concept": "formula:ncfo,capex",
                "source_url": f"{ncfo['source_url']};{capex['source_url']}",
                "source_sha256": f"{ncfo['source_sha256']};{capex['source_sha256']}",
            }
            break
    return output, manifests


def evidence_fact(row: dict[str, Any]) -> dict[str, str]:
    return {
        "field": str(row["field"]),
        "value_text": str(row["value_text"]),
        "period_start": str(row["period_start"]),
        "period_end": str(row["period_end"]),
        "filed": str(row["filed"]),
        "accession": str(row["accession"]),
        "derivation": str(row["derivation"]),
        "status": str(row["status"]),
        "source_concept": str(row["source_concept"]),
        "source_url": str(row["source_url"]),
        "source_sha256": str(row["source_sha256"]),
    }


def build(benchmark: Path, output: Path, fact_runs: tuple[Path, ...] = FACT_RUNS) -> dict[str, Any]:
    cohort, benchmark_manifest_sha256, cohort_sha256 = verify_benchmark(benchmark)
    members = list(cohort.get("members") or [])
    if cohort.get("cohort_size") != EXPECTED_COHORT_SIZE or len(members) != EXPECTED_COHORT_SIZE:
        raise RuntimeError("validated SEC cohort is not exactly 1,000 members")
    if cohort.get("cutoff") != EXPECTED_CUTOFF:
        raise RuntimeError("validated SEC cohort cutoff changed")
    ciks = {str(row["cik"]).zfill(10) for row in members}
    if len(ciks) != EXPECTED_COHORT_SIZE:
        raise RuntimeError("validated SEC cohort contains duplicate CIKs")
    profitability, fact_manifests = load_latest_profitability(fact_runs, ciks)

    companies: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for member in sorted(members, key=lambda row: int(row["rank"])):
        cik = str(member["cik"]).zfill(10)
        sic = str(member.get("sic") or "")
        accounting_regime = "financial_or_utility" if is_financial_or_utility(sic) else "ordinary_operating"
        profitability_field = "netinc" if accounting_regime == "financial_or_utility" else "fcf"
        observation = profitability.get((cik, profitability_field))
        if observation is None:
            missing.append(
                {
                    "cik": cik,
                    "ticker": str(member["sec_ticker"]),
                    "required_field": profitability_field,
                }
            )
            continue
        revenue = Decimal(str(member["ttm_revenue"]))
        selected_profitability = Decimal(str(observation["value_text"]))
        if revenue <= 0:
            raise RuntimeError(f"nonpositive cohort revenue for {member['sec_ticker']}")
        companies.append(
            {
                "cik": cik,
                "ticker": str(member["sec_ticker"]).upper(),
                "company_name": str(member["name"]),
                "sec_sic": sic,
                "exchange": str(member["exchange"]),
                "accounting_regime": accounting_regime,
                "profitability_factor": (
                    "trailing_four_quarter_net_income"
                    if profitability_field == "netinc"
                    else "trailing_four_quarter_free_cash_flow"
                ),
                "ttm_revenue": decimal_text(revenue),
                "selected_profitability": decimal_text(selected_profitability),
                "revenue_rank": int(member["rank"]),
                "fundamental_evidence": {
                    "revenue": {
                        "period_end": str(member["reportperiod"]),
                        "quarters": member["quarters"],
                        "derivation": "sum latest four canonical SEC ARQ revenue facts",
                    },
                    "selected_profitability": evidence_fact(observation),
                },
            }
        )
    if missing:
        raise RuntimeError(
            "routed SEC profitability is missing for "
            + json.dumps(missing, sort_keys=True, separators=(",", ":"))
        )
    if len(companies) != EXPECTED_COHORT_SIZE:
        raise RuntimeError(f"expected 1,000 complete companies, found {len(companies)}")

    with localcontext() as context:
        context.prec = 60
        profits = [Decimal(row["selected_profitability"]) for row in companies]
        mean = sum(profits) / Decimal(len(profits))
        variance = sum((value - mean) ** 2 for value in profits) / Decimal(len(profits))
        standard_deviation = context.sqrt(variance)
        if standard_deviation == 0:
            raise RuntimeError("selected profitability standard deviation is zero")
        for row in companies:
            z_score = (Decimal(row["selected_profitability"]) - mean) / standard_deviation
            multiplier = context.exp(Decimal("0.03") * z_score)
            adjusted_scale = Decimal(row["ttm_revenue"]) * multiplier
            row["profitability_z_score"] = decimal_text(z_score)
            row["profitability_multiplier"] = decimal_text(multiplier)
            row["adjusted_scale"] = decimal_text(adjusted_scale)

    artifact = {
        "schema": "corbanu.sec-fundamental-universe.v1",
        "universe_id": "sec-registered-us-listed-top1000-4q-revenue-v81-20260828",
        "as_of_date": EXPECTED_CUTOFF,
        "company_count": len(companies),
        "source": {
            "provider": "U.S. Securities and Exchange Commission",
            "sec_only": True,
            "methodology_version": PROFITABILITY_METHODOLOGY,
            "membership_methodology_version": MEMBERSHIP_METHODOLOGY,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "cohort_sha256": cohort_sha256,
            "cohort_fingerprint": str(cohort["cohort_fingerprint"]),
            "companyfacts_archive_sha256": COMPANYFACTS_ARCHIVE_SHA256,
            "submissions_archive_sha256": SUBMISSIONS_ARCHIVE_SHA256,
            "fact_run_manifests": fact_manifests,
        },
        "methodology": {
            "selection": "top 1,000 eligible USD-reporting U.S.-listed SEC registrants by positive trailing-four-quarter revenue",
            "membership": "frozen validated v77 SEC cohort; descending SEC TTM revenue then CIK",
            "profitability": "v80 SEC capex conventions plus v81 net-income taxonomy overlay and reviewed v78 filing extensions",
            "ttm_revenue": "sum of the latest four canonical SEC ARQ revenue facts",
            "profitability_router": "SEC SIC 4900-4999 and 6000-6799 use TTM net income; all other issuers use TTM free cash flow",
            "profitability_overlay": "TTM revenue multiplied by exp(0.03 * population z-score of selected profitability)",
            "licensed_vendor_inputs": False,
        },
        "companies": companies,
    }
    artifact["universe_sha256"] = sha256(canonical_bytes(artifact))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(artifact) + b"\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = build(args.benchmark, args.output)
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
