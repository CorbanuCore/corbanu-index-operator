#!/usr/bin/env python3
"""Local Corbanu Indexes execution service.

The public API accepts only a frozen index request.  It cannot execute shell
commands or select arbitrary inference endpoints.  Jobs run against two local
H200 tunnels and become valid only when every raw scoring response matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_JOBS_DIR = Path("/home/postfiat/var/corbanu-indexes/jobs")
DEFAULT_UNIVERSE = ROOT / "data" / "sec_top1000_universe_20260827.json"
PRIMARY_ENDPOINT = os.environ.get("CORBANU_PRIMARY_ENDPOINT", "http://127.0.0.1:18001")
REPLAY_ENDPOINT = os.environ.get("CORBANU_REPLAY_ENDPOINT", "http://127.0.0.1:18002")
MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MODEL_PROFILE = "qwen3.8-27b-fp8-h200-sglang-deterministic-noradix-schema-v1"
ALLOWED_ORIGINS = {
    "https://corbanu.com",
    "https://www.corbanu.com",
    "http://127.0.0.1:4000",
    "http://localhost:4000",
}
ANCHORS = (0, 25, 50, 75, 100)
WEIGHT_UNITS = 1_000_000_000_000
HOLDING_CAP = Decimal("0.20")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_bytes(canonical_bytes(payload) + b"\n")
    os.replace(temporary, path)


def parse_decimal(value: Any) -> Decimal:
    text = str(value or "0").replace(",", "").strip()
    return Decimal(text or "0")


def load_universe(universe_path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    artifact = json.loads(universe_path.read_bytes())
    if artifact.get("schema") != "corbanu.sec-fundamental-universe.v1":
        raise RuntimeError("unsupported universe schema")
    expected_hash = str(artifact.get("universe_sha256") or "")
    hash_payload = dict(artifact)
    hash_payload.pop("universe_sha256", None)
    if len(expected_hash) != 64 or sha256(canonical_bytes(hash_payload)) != expected_hash:
        raise RuntimeError("universe content hash does not verify")
    rows = list(artifact.get("companies") or [])
    if artifact.get("company_count") != 1000 or len(rows) != 1000:
        raise RuntimeError(f"canonical universe must contain exactly 1,000 companies; found {len(rows)}")
    if len({str(row.get("cik")) for row in rows}) != 1000:
        raise RuntimeError("canonical universe contains duplicate CIKs")
    required = {
        "cik", "ticker", "company_name", "accounting_regime", "profitability_factor",
        "ttm_revenue", "selected_profitability", "profitability_z_score",
        "profitability_multiplier", "adjusted_scale",
    }
    for row in rows:
        if not required.issubset(row):
            raise RuntimeError(f"universe row is incomplete for {row.get('ticker', 'unknown')}")
        if parse_decimal(row["adjusted_scale"]) <= 0:
            raise RuntimeError(f"universe adjusted scale is invalid for {row['ticker']}")
    rows.sort(key=lambda row: (str(row["ticker"]), str(row["cik"])))
    metadata = {
        "id": artifact["universe_id"],
        "company_count": len(rows),
        "as_of_date": artifact["as_of_date"],
        "universe_sha256": expected_hash,
        "source": artifact["source"],
        "methodology": artifact["methodology"],
    }
    return rows, expected_hash, metadata


RUBRIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "basket_description": {"type": "string"},
        "scoring_heuristic": {"type": "string"},
        "score_0": {"type": "string"},
        "score_25": {"type": "string"},
        "score_50": {"type": "string"},
        "score_75": {"type": "string"},
        "score_100": {"type": "string"},
    },
    "required": [
        "basket_description",
        "scoring_heuristic",
        "score_0",
        "score_25",
        "score_50",
        "score_75",
        "score_100",
    ],
}

SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticker": {"type": "string"},
        "score": {"type": "integer", "enum": list(ANCHORS)},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning_block": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "string"},
        },
    },
    "required": ["ticker", "score", "confidence", "reasoning_block"],
}


def chat_request(messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }


def rubric_request(index_request: dict[str, Any]) -> dict[str, Any]:
    mandate = index_request["mandate"]
    return chat_request(
        [
            {
                "role": "system",
                "content": (
                    "You are the deterministic mandate compiler for a thematic equity index. "
                    "Convert the supplied title and mandate into an explicit qualitative scoring rubric. "
                    "Use only durable knowledge an LLM can know from training data. The downstream scorer "
                    "cannot retrieve documents. Preserve the user's intended exposure and exclusions. "
                    "The levels must mean: 0 no meaningful exposure; 25 incidental exposure; 50 material "
                    "but not a major strategic pillar; 75 a major strategic pillar with direct exposure; "
                    "100 a pure-play or defining exposure. Return only JSON matching the schema."
                ),
            },
            {
                "role": "user",
                "content": f"Index title: {mandate['title']}\nIndex mandate: {mandate['phrase']}",
            },
        ],
        "thematic_index_rubric",
        RUBRIC_SCHEMA,
        1800,
    )


def score_request(index_request: dict[str, Any], rubric: dict[str, Any], company: dict[str, Any]) -> dict[str, Any]:
    input_packet = {
        "index_title": index_request["mandate"]["title"],
        "index_mandate": index_request["mandate"]["phrase"],
        "locked_scoring_rubric": rubric,
        "company": {
            "ticker": company["ticker"],
            "company_name": company["company_name"],
            "sec_cik": company["cik"],
            "sec_sic": company.get("sec_sic"),
            "sec_sic_description": company.get("sec_sic_description"),
        },
        "supplemental_transcript_information": {
            "provided": False,
            "text": None,
            "instruction": "No supplemental transcript is supplied for this run. Do not invent one.",
        },
    }
    return chat_request(
        [
            {
                "role": "system",
                "content": (
                    "You are the deterministic company scorer for a thematic equity index. Apply EXACTLY "
                    "the locked scoring rubric supplied by the user. Use the company's widely known identity, "
                    "products, services, technologies, industry role, and strategic direction from your training "
                    "data. Supplemental transcript text, when supplied, grounds and updates that knowledge; when "
                    "it is absent, score from durable training knowledge and state uncertainty honestly. Financial "
                    "size and profitability do not determine thematic relevance and must not affect the score. "
                    "Return the ticker exactly, one anchor score (0, 25, 50, 75, or 100), a confidence from 0 to "
                    "100, and exactly three substantive reasoning paragraphs. Return only JSON matching the schema."
                ),
            },
            {"role": "user", "content": json.dumps(input_packet, sort_keys=True, ensure_ascii=False)},
        ],
        "thematic_company_score",
        SCORE_SCHEMA,
        1800,
    )


@dataclass(frozen=True)
class Completion:
    raw: str
    parsed: dict[str, Any]
    request_sha256: str
    content_sha256: str


def complete(endpoint: str, payload: dict[str, Any], timeout: int = 300) -> Completion:
    encoded = canonical_bytes(payload)
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                envelope = json.loads(response.read())
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"model completion ended with {choice.get('finish_reason')}")
            raw = choice["message"]["content"]
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("model response is not a JSON object")
            return Completion(raw, parsed, sha256(encoded), sha256(raw.encode("utf-8")))
        except (OSError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"inference request failed after retries: {last_error}")


def validate_rubric(value: dict[str, Any]) -> None:
    if set(value) != set(RUBRIC_SCHEMA["required"]):
        raise ValueError("rubric fields do not match the locked schema")
    if any(len(str(value[key]).strip()) < 20 for key in RUBRIC_SCHEMA["required"]):
        raise ValueError("rubric contains an underspecified field")


def validate_score(value: dict[str, Any], ticker: str) -> None:
    if set(value) != {"ticker", "score", "confidence", "reasoning_block"}:
        raise ValueError(f"score fields invalid for {ticker}")
    if str(value["ticker"]).upper() != ticker:
        raise ValueError(f"ticker mismatch for {ticker}")
    if value["score"] not in ANCHORS:
        raise ValueError(f"score is not an anchor for {ticker}")
    if not isinstance(value["confidence"], int) or not 0 <= value["confidence"] <= 100:
        raise ValueError(f"confidence invalid for {ticker}")
    paragraphs = value["reasoning_block"]
    if not isinstance(paragraphs, list) or len(paragraphs) != 3 or any(len(str(p)) < 80 for p in paragraphs):
        raise ValueError(f"reasoning block invalid for {ticker}")


def capped_weights(rows: list[dict[str, Any]], raw_field: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    with localcontext() as context:
        context.prec = 60
        raw = {row["cik"]: Decimal(str(row[raw_field])) for row in rows}
        active = set(raw)
        fixed: dict[str, Decimal] = {}
        remaining = Decimal(1)
        while active:
            total = sum(raw[key] for key in active)
            if total <= 0:
                equal = remaining / Decimal(len(active))
                for key in active:
                    fixed[key] = equal
                break
            proposed = {key: remaining * raw[key] / total for key in active}
            breached = sorted(key for key, weight in proposed.items() if weight > HOLDING_CAP)
            if not breached:
                fixed.update(proposed)
                break
            for key in breached:
                fixed[key] = HOLDING_CAP
                active.remove(key)
                remaining -= HOLDING_CAP
        exact_units = {key: fixed[key] * Decimal(WEIGHT_UNITS) for key in fixed}
        units = {key: int(value.to_integral_value(rounding=ROUND_FLOOR)) for key, value in exact_units.items()}
        residual = WEIGHT_UNITS - sum(units.values())
        order = sorted(units, key=lambda key: (-(exact_units[key] - units[key]), key))
        for key in order[:residual]:
            units[key] += 1
    output = []
    for row in rows:
        copy = dict(row)
        copy["weight_units"] = units[row["cik"]]
        copy["weight"] = f"{Decimal(units[row['cik']]) / Decimal(WEIGHT_UNITS):.12f}"
        copy["weight_percent"] = f"{Decimal(units[row['cik']]) / Decimal(10_000_000_000):.6f}"
        output.append(copy)
    return sorted(output, key=lambda row: (-row["weight_units"], row["ticker"], row["cik"]))


def build_weights(
    index_request: dict[str, Any], universe: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if index_request["construction"]["weighting"] != "fundamental":
        raise ValueError("The public local operator currently runs the recommended fundamental method only.")
    cutoff = int(index_request["classification"]["relevance_cutoff"])
    score_by_cik = {row["cik"]: row for row in scores}
    eligible = []
    for company in universe:
        result = score_by_cik[company["cik"]]
        if int(result["score"]) < cutoff:
            continue
        score_multiplier = Decimal(int(result["score"])) / Decimal(100)
        adjusted_scale = Decimal(company["adjusted_scale"])
        row = {
            "cik": company["cik"],
            "ticker": company["ticker"],
            "company_name": company["company_name"],
            "score": int(result["score"]),
            "confidence": int(result["confidence"]),
            "reasoning_block": result["reasoning_block"],
            "accounting_regime": company["accounting_regime"],
            "profitability_factor": company["profitability_factor"],
            "ttm_revenue": company["ttm_revenue"],
            "selected_profitability": company["selected_profitability"],
            "profitability_z_score": company["profitability_z_score"],
            "profitability_multiplier": company["profitability_multiplier"],
            "adjusted_scale": company["adjusted_scale"],
            "score_multiplier": str(score_multiplier),
            "raw_weight": str(adjusted_scale * score_multiplier),
        }
        eligible.append(row)
    return capped_weights(eligible, "raw_weight")


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("request"), dict):
        raise ValueError("request envelope is missing request")
    request = payload["request"]
    if request.get("schema") != "corbanu.index-request.v1":
        raise ValueError("unsupported request schema")
    mandate = request.get("mandate") or {}
    title = re.sub(r"\s+", " ", str(mandate.get("title") or "")).strip()
    phrase = re.sub(r"\s+", " ", str(mandate.get("phrase") or "")).strip()
    if not 3 <= len(title) <= 80:
        raise ValueError("index title must be 3 through 80 characters")
    if not 12 <= len(phrase) <= 600:
        raise ValueError("index mandate must be 12 through 600 characters")
    if request.get("model", {}).get("id") != MODEL_ID or request.get("model", {}).get("revision") != MODEL_REVISION:
        raise ValueError("the local operator currently accepts only the admitted Qwen model revision")
    if request.get("model", {}).get("execution_profile") != MODEL_PROFILE:
        raise ValueError("model execution profile is not admitted")
    cutoff = request.get("classification", {}).get("relevance_cutoff")
    if not isinstance(cutoff, int) or not 0 <= cutoff <= 100:
        raise ValueError("relevance cutoff must be an integer from 0 to 100")
    if request.get("construction", {}).get("weighting") != "fundamental":
        raise ValueError("cap weighting is not enabled in the public operator; choose Fundamental weighted")
    request["mandate"] = {"title": title, "phrase": phrase}
    return request


class JobStore:
    def __init__(self, jobs_dir: Path, universe_path: Path, workers: int) -> None:
        self.jobs_dir = jobs_dir
        self.universe, self.universe_sha256, self.universe_metadata = load_universe(universe_path)
        self.workers = workers
        self.pending: queue.Queue[str] = queue.Queue(maxsize=3)
        self.lock = threading.Lock()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="corbanu-index-worker")
        self.worker.start()

    def path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "job.json"

    def artifact_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "artifact.json"

    def read(self, job_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"idx_[0-9a-f]{16}", job_id):
            return None
        path = self.path(job_id)
        return json.loads(path.read_text()) if path.exists() else None

    def update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            state = self.read(job_id) or {"job_id": job_id}
            state.update(fields)
            state["updated_at_unix"] = int(time.time())
            atomic_json(self.path(job_id), state)

    def submit(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        requested_universe = request.get("universe") or {}
        if requested_universe.get("id") != self.universe_metadata["id"]:
            raise ValueError("request universe id does not match the operator's frozen universe")
        if requested_universe.get("universe_sha256") != self.universe_sha256:
            raise ValueError("request universe hash does not match the operator's frozen universe")
        request_hash = sha256(
            canonical_bytes({"request": request, "universe_sha256": self.universe_sha256})
        )
        job_id = f"idx_{request_hash[:16]}"
        existing = self.read(job_id)
        if existing:
            if existing.get("status") == "failed":
                existing.update(
                    {
                        "status": "queued",
                        "stage": "request_frozen",
                        "progress": {"completed": 0, "total": len(self.universe) * 2 + 2},
                    }
                )
                existing.pop("error", None)
                atomic_json(self.path(job_id), existing)
                self.pending.put_nowait(job_id)
                return existing, True
            return existing, False
        if self.pending.full():
            raise RuntimeError("The public operator queue is full. Try again after the current runs finish.")
        state = {
            "job_id": job_id,
            "request_sha256": request_hash,
            "status": "queued",
            "stage": "request_frozen",
            "progress": {"completed": 0, "total": len(self.universe) * 2 + 2},
            "created_at_unix": int(time.time()),
            "updated_at_unix": int(time.time()),
            "request": request,
            "universe": self.universe_metadata,
        }
        atomic_json(self.path(job_id), state)
        self.pending.put_nowait(job_id)
        return state, True

    def public_state(self, state: dict[str, Any], include_artifact: bool = False) -> dict[str, Any]:
        keys = (
            "job_id", "request_sha256", "status", "stage", "progress", "created_at_unix",
            "updated_at_unix", "universe", "error", "result_summary", "artifact_sha256",
        )
        public = {key: state[key] for key in keys if key in state}
        if include_artifact and state.get("status") == "valid" and self.artifact_path(state["job_id"]).exists():
            public["artifact"] = json.loads(self.artifact_path(state["job_id"]).read_text())
        return public

    def _worker_loop(self) -> None:
        while True:
            job_id = self.pending.get()
            try:
                self.run(job_id)
            except Exception as error:  # noqa: BLE001
                self.update(job_id, status="failed", stage="failed", error=str(error)[:1000])
            finally:
                self.pending.task_done()

    def run(self, job_id: str) -> None:
        state = self.read(job_id)
        if not state:
            raise RuntimeError("job disappeared")
        request = state["request"]
        total = len(self.universe) * 2 + 2
        completed = 0
        self.update(job_id, status="running", stage="compiling_mandate", progress={"completed": completed, "total": total})

        rubric_payload = rubric_request(request)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_primary = pool.submit(complete, PRIMARY_ENDPOINT, rubric_payload)
            future_replay = pool.submit(complete, REPLAY_ENDPOINT, rubric_payload)
            primary_rubric = future_primary.result()
            replay_rubric = future_replay.result()
        validate_rubric(primary_rubric.parsed)
        validate_rubric(replay_rubric.parsed)
        if primary_rubric.raw.encode("utf-8") != replay_rubric.raw.encode("utf-8"):
            raise RuntimeError("mandate compiler replay was not byte-identical")
        completed += 2
        self.update(job_id, stage="scoring_primary", progress={"completed": completed, "total": total})

        results: dict[str, dict[str, Any]] = {}

        def score_one(endpoint: str, host: str, company: dict[str, Any]) -> tuple[str, str, Completion]:
            payload = score_request(request, primary_rubric.parsed, company)
            response = complete(endpoint, payload)
            validate_score(response.parsed, company["ticker"])
            return host, company["cik"], response

        # Run each machine as its own ordered pass.  This preserves the proven
        # 32-request H200 profile without allowing timing on one verifier to
        # alter the other verifier's batch schedule.
        for endpoint, host in ((PRIMARY_ENDPOINT, "primary"), (REPLAY_ENDPOINT, "replay")):
            self.update(job_id, stage=f"scoring_{host}", progress={"completed": completed, "total": total})
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(score_one, endpoint, host, company) for company in self.universe]
                for future in as_completed(futures):
                    result_host, cik, response = future.result()
                    results.setdefault(cik, {})[result_host] = {
                        "raw": response.raw,
                        "parsed": response.parsed,
                        "request_sha256": response.request_sha256,
                        "content_sha256": response.content_sha256,
                    }
                    completed += 1
                    if completed % 8 == 0 or completed == total:
                        self.update(
                            job_id,
                            stage=f"scoring_{host}",
                            progress={"completed": completed, "total": total},
                        )

        score_rows = []
        comparisons = []
        for company in self.universe:
            pair = results.get(company["cik"], {})
            if set(pair) != {"primary", "replay"}:
                raise RuntimeError(f"missing replay pair for {company['ticker']}")
            primary = pair["primary"]
            replay = pair["replay"]
            byte_identical = primary["raw"].encode("utf-8") == replay["raw"].encode("utf-8")
            request_identical = primary["request_sha256"] == replay["request_sha256"]
            comparisons.append(
                {
                    "cik": company["cik"],
                    "ticker": company["ticker"],
                    "request_sha256": primary["request_sha256"],
                    "primary_content_sha256": primary["content_sha256"],
                    "replay_content_sha256": replay["content_sha256"],
                    "byte_identical": byte_identical,
                    "request_identical": request_identical,
                }
            )
            if not byte_identical or not request_identical:
                raise RuntimeError(f"byte replay failed for {company['ticker']}")
            score_rows.append({"cik": company["cik"], **primary["parsed"]})

        self.update(job_id, stage="constructing_weights", progress={"completed": total, "total": total})
        holdings = build_weights(request, self.universe, score_rows)
        if not holdings:
            raise RuntimeError("no companies met the requested relevance cutoff")
        score_vector = [
            {"cik": row["cik"], "ticker": row["ticker"], "score": row["score"], "confidence": row["confidence"]}
            for row in sorted(score_rows, key=lambda row: (row["ticker"], row["cik"]))
        ]
        replay_proof = {
            "comparison_count": len(comparisons),
            "byte_identical_count": sum(int(row["byte_identical"]) for row in comparisons),
            "request_identical_count": sum(int(row["request_identical"]) for row in comparisons),
            "rubric_primary_content_sha256": primary_rubric.content_sha256,
            "rubric_replay_content_sha256": replay_rubric.content_sha256,
            "rubric_byte_identical": primary_rubric.raw.encode("utf-8") == replay_rubric.raw.encode("utf-8"),
            "comparisons_sha256": sha256(canonical_bytes(comparisons)),
        }
        stable_artifact = {
            "schema": "corbanu.index-artifact.v1",
            "valid": True,
            "job_id": job_id,
            "request_sha256": state["request_sha256"],
            "index_title": request["mandate"]["title"],
            "index_mandate": request["mandate"]["phrase"],
            "rubric": primary_rubric.parsed,
            "rubric_raw_utf8": primary_rubric.raw,
            "rubric_request_sha256": primary_rubric.request_sha256,
            "rubric_content_sha256": primary_rubric.content_sha256,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "execution_profile": MODEL_PROFILE},
            "universe": state["universe"],
            "construction": {
                "weighting": "fundamental",
                "formula": "score_multiplier * TTM revenue * exp(0.03 * selected profitability population z-score)",
                "profitability_router": "TTM net income for financials and utilities; TTM free cash flow otherwise",
                "relevance_cutoff": request["classification"]["relevance_cutoff"],
                "holding_cap": "0.20",
                "weight_units": WEIGHT_UNITS,
            },
            "score_vector_sha256": sha256(canonical_bytes(score_vector)),
            "scores": sorted(score_rows, key=lambda row: (row["ticker"], row["cik"])),
            "replay_proof": replay_proof,
            "replay_comparisons": comparisons,
            "constituent_count": len(holdings),
            "holdings": holdings,
        }
        artifact_hash = sha256(canonical_bytes(stable_artifact))
        artifact = {**stable_artifact, "artifact_sha256": artifact_hash}
        atomic_json(self.artifact_path(job_id), artifact)
        summary = {
            "index_title": artifact["index_title"],
            "constituent_count": len(holdings),
            "top_holdings": [
                {key: row[key] for key in ("ticker", "company_name", "score", "confidence", "weight", "weight_percent")}
                for row in holdings[:20]
            ],
            "byte_replay": f"{replay_proof['byte_identical_count']}/{replay_proof['comparison_count']}",
        }
        self.update(
            job_id,
            status="valid",
            stage="valid",
            progress={"completed": total, "total": total},
            artifact_sha256=artifact_hash,
            result_summary=summary,
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "CorbanuIndexOperator/0.1"

    @property
    def store(self) -> JobStore:
        return self.server.store  # type: ignore[attr-defined]

    def origin(self) -> str | None:
        value = self.headers.get("Origin")
        return value if value in ALLOWED_ORIGINS else None

    def send_json(self, status: int, payload: Any) -> None:
        encoded = canonical_bytes(payload) + b"\n"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self.origin():
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "origin not allowed"})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", self.origin() or "")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {"ok": True, "service": "corbanu-index-operator", "universe_count": len(self.store.universe)},
            )
            return
        match = re.fullmatch(r"/v1/indexes/(idx_[0-9a-f]{16})(/artifact)?", path)
        if not match:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        state = self.store.read(match.group(1))
        if not state:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
            return
        if match.group(2):
            artifact_path = self.store.artifact_path(match.group(1))
            if state.get("status") != "valid" or not artifact_path.exists():
                self.send_json(HTTPStatus.CONFLICT, {"error": "artifact is not valid yet"})
                return
            self.send_json(HTTPStatus.OK, json.loads(artifact_path.read_text()))
            return
        self.send_json(HTTPStatus.OK, self.store.public_state(state))

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/v1/indexes":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.headers.get("Origin") and not self.origin():
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "origin not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32_768:
                raise ValueError("request body must be between 1 byte and 32 KiB")
            payload = json.loads(self.rfile.read(length))
            request = validate_request(payload)
            state, created = self.store.submit(request)
            self.send_json(
                HTTPStatus.ACCEPTED if created else HTTPStatus.OK,
                {**self.store.public_state(state), "created": created},
            )
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self.send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
        except json.JSONDecodeError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "request body is not valid JSON"})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Corbanu Indexes API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--workers-per-host", type=int, default=32)
    args = parser.parse_args()
    store = JobStore(args.jobs_dir, args.universe, args.workers_per_host)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = store  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "service": "corbanu-index-operator",
                "listen": f"http://{args.host}:{args.port}",
                "universe_count": len(store.universe),
                "primary": PRIMARY_ENDPOINT,
                "replay": REPLAY_ENDPOINT,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
