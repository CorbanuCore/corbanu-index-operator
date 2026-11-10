"""Stream native stage receipts to disk while the frozen index is generated."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from evidence import save
from native_contract import CANARIES, Contract, receipt_path
from replay_contract import core as c
from replay_contract.execute import seal

ROOT = Path(__file__).resolve().parent


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True)
    p.add_argument("--mode", choices=["canary", "full"], required=True)
    p.add_argument("--copies", type=int, choices=[1, 2], default=1)
    return p.parse_args()


def main(dp_size, rank):
    a = args()
    contract = Contract()
    profile = contract.profile
    host = c.load(ROOT / "host-identity.json")
    directory = ROOT / "evidence" / a.round
    directory.mkdir(parents=True, exist_ok=True)
    chosen = (list(contract.rows) if a.mode == "full" else
              [next(s for s, r in contract.rows.items() if r["context"]["ticker"] == t) for t in CANARIES])
    mine = [sid for i, sid in enumerate(chosen) if i % dp_size == rank]
    assert mine
    for sid in mine:
        assert not receipt_path(directory, sid, 0, "first_pass").exists(), "Completed observations cannot be rerun in place"
    from examples._config import create_llm, setup_common_env
    setup_common_env(model_name="/workspace/model", sampler_force_topk=4096,
                     set_override_experts=False, enable_eplb=False)
    started = time.monotonic()
    llm = create_llm("/workspace/model", rank, ep_size=4, cp_size=1, data_parallel_size=4,
                     max_num_batched_tokens=16384, max_model_len=65536, max_num_seqs=2,
                     gpu_memory_utilization=0.85, lookahead_tokens=0)
    startup = time.monotonic() - started
    from deepinfer.sampling_params import SamplingParams
    from deepinfer._C.engine import RequestOutput
    total_tokens, total_complete = 0, 0

    def generate(tasks, stage, diagnostic=False):
        nonlocal total_tokens, total_complete
        active = {}
        began = time.monotonic()
        for task in tasks:
            sid, copy, notes, parent = task
            request = contract.request(sid, stage, notes)
            sp = SamplingParams(**{**request["effective_sampling"], "temperature": 0})
            assert {k: getattr(sp, k) for k in request["effective_sampling"]} == request["effective_sampling"]
            sequence = llm._add_request(None, sp, request["input_token_ids"], None)
            active[sequence.request_id] = {"task": task, "ids": [], "first_token_at": None}
        next_tasks, completed, last_saved = [], [], 0
        while True:
            unfinished = llm.llm_engine.has_unfinished_requests()
            outputs = (llm.llm_engine.step() if unfinished else llm.llm_engine.flush_inflight_reqs())
            for raw_output in outputs:
                # step() returns pooled native handles, exactly as in LLM._run_engine.
                output = RequestOutput.clone(raw_output)
                item = active[output.request_id]
                if item["first_token_at"] is None:
                    item["first_token_at"] = time.monotonic()
                item["ids"].append(int(output.token_id))
                total_tokens += 1
                assert total_tokens <= 750000, "Per-rank output-token ceiling reached"
                if output.finished:
                    sid, copy, notes, parent = item["task"]
                    metadata = {"host": host, "rank": rank, "pid": os.getpid(), "round": a.round,
                                "copy": copy, "startup_seconds": startup,
                                "seconds_from_stage_start": time.monotonic() - began,
                                "seconds_from_first_token": time.monotonic() - item["first_token_at"],
                                "finished_at_ns": time.time_ns(), "diagnostic_only": diagnostic}
                    receipt, raw = contract.observe(sid, stage, notes, parent, item["ids"], output.finish_reason, metadata)
                    path = (directory / "stress" / f"rank{rank}-copy{copy}.json" if diagnostic
                            else receipt_path(directory, sid, copy, stage))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.with_suffix(".bin").open("xb") as stream:
                        stream.write(raw)
                    seal(path, receipt)
                    decision = receipt["transition"]
                    assert decision["action"] != "failed", (sid, stage, decision)
                    if decision["action"] == "next":
                        next_tasks.append((sid, copy, decision["notes"], receipt["commitment"]))
                    completed.append(receipt)
                    total_complete += 1
                    print(json.dumps({"rank": rank, "round": a.round, "security_id": sid,
                                      "stage": stage, "tokens": len(item["ids"]),
                                      "decision": decision["action"], "total_complete": total_complete}), flush=True)
                    del active[output.request_id]
            now = time.monotonic()
            if now - last_saved > 5 or not unfinished:
                save(directory / f"progress-rank{rank}.json", {
                    "at": time.time(), "rank": rank, "stage": stage, "completed_stages": total_complete,
                    "output_tokens": total_tokens, "active_requests": len(active),
                    "seconds_since_process_start": now - started,
                    "partial_outputs": [{"security_id": v["task"][0], "copy": v["task"][1],
                                         "output_token_ids": v["ids"]} for v in active.values() if v["ids"]],
                })
                last_saved = now
            if not unfinished:
                break
        assert not active, "Unfinished or missing request outputs"
        return next_tasks, completed

    try:
        tasks = [(sid, copy, "", None) for sid in mine for copy in range(a.copies)]
        all_completed = []
        for stage in c.STAGES:
            tasks, completed = generate(tasks, stage)
            all_completed.extend(completed)
        assert not tasks
        seal(directory / f"complete-rank{rank}.json", {"security_ids": mine, "copies": a.copies,
             "rank": rank, "pid": os.getpid(), "host": host, "round": a.round,
             "stage_count_including_diagnostics": total_complete, "output_tokens": total_tokens,
             "elapsed_seconds": time.monotonic() - started, "startup_seconds": startup})

        # Keep the qualified model in GPU memory after its replay smoke check.
        # Commands use the already-frozen job; no live index is published.
        command_root = ROOT / "commands"
        command_root.mkdir(exist_ok=True)
        seen = set()
        while time.time() < host["deadline"] - 30 and not (ROOT / "stop").exists():
            save(ROOT / "evidence" / f"ready-rank{rank}.json", {
                "state": "ready", "heartbeat": time.time(), "rank": rank,
                "pid": os.getpid(), "native_profile_sha256": contract.profile_hash,
                "model_revision": profile["model_revision"],
                "startup_seconds": startup, "completed_stages": total_complete,
                "deadline": host["deadline"], "boot_round": "boot-smoke"})
            pending = [p for p in sorted(command_root.glob("*.json")) if p.name not in seen]
            if not pending:
                time.sleep(2)
                continue
            command_path = pending[0]
            command = c.load(command_path)
            assert set(command) == {"round", "mode", "copies"}, "Invalid command schema"
            name = command["round"]
            assert isinstance(name, str) and 1 <= len(name) <= 80
            assert all(x.isalnum() or x in "_-" for x in name), "Invalid round name"
            assert name != "boot-smoke"
            assert command["mode"] in ("canary", "full")
            assert type(command["copies"]) is int and command["copies"] in (1, 2)
            a.round, a.mode, a.copies = name, command["mode"], command["copies"]
            directory = ROOT / "evidence" / name
            directory.mkdir(parents=True, exist_ok=True)
            chosen = (list(contract.rows) if a.mode == "full" else
                      [next(s for s,r in contract.rows.items() if r["context"]["ticker"] == t)
                       for t in CANARIES])
            mine = [sid for i,sid in enumerate(chosen) if i % dp_size == rank]
            assert mine
            for sid in mine:
                for copy in range(a.copies):
                    assert not receipt_path(directory,sid,copy,"first_pass").exists()
            save(ROOT / "evidence" / f"ready-rank{rank}.json", {
                "state": "busy", "heartbeat": time.time(), "rank": rank,
                "pid": os.getpid(), "round": name, "deadline": host["deadline"]})
            tasks = [(sid,copy,"",None) for sid in mine for copy in range(a.copies)]
            before = total_complete
            for stage in c.STAGES:
                tasks, completed = generate(tasks,stage)
            assert not tasks
            seal(directory / f"complete-rank{rank}.json", {
                "security_ids": mine, "copies": a.copies, "rank": rank,
                "pid": os.getpid(), "host": host, "round": name,
                "stage_count": total_complete-before, "output_tokens": total_tokens,
                "native_profile_sha256": contract.profile_hash})
            seen.add(command_path.name)
    finally:
        llm.close()


if __name__ == "__main__":
    sys.path.insert(0, "/workspace/DeepInfer")
    from examples._launch import launch_dp
    launch_dp(main, 4)
