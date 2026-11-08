"""Warm DeepSeek worker: boot replay check, then a question loop that never exits on a bad request.

Questions arrive as JSON files in /workspace/trial/commands ({"round", "mode": "question", "question"}).
Every rank answers every question with the pinned profile; receipts go to evidence/questions/<round>/.
A failed or malformed command writes evidence/command-errors/<file>-rank<r>.json and the engine stays up.
The lease deadline is re-read from lease.json each loop so the controller can extend it.
"""
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
    p.add_argument("--resume", action="store_true")
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
    for sid in ([] if a.resume else mine):
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


    QUESTION_MAX_TOKENS = 16384
    SYSTEM_PROMPT = "You are a helpful assistant."

    def deadline():
        lease = ROOT / "lease.json"
        try:
            return float(c.load(lease)["deadline"]) if lease.exists() else float(host["deadline"])
        except Exception:
            return float(host["deadline"])

    def generate_question(command):
        import base64
        import hashlib
        from native_contract import SPECIAL
        from replay_contract.runtime import channels
        nonlocal total_tokens, total_complete
        if set(command) != {"round", "mode", "question"}:
            raise ValueError("question command needs exactly round, mode, question")
        name, question = command["round"], command["question"]
        if not (isinstance(name, str) and 1 <= len(name) <= 80 and all(x.isalnum() or x in "_-" for x in name)):
            raise ValueError("invalid round name")
        if not (isinstance(question, str) and 1 <= len(question) <= 20000):
            raise ValueError("question must be 1-20000 characters")
        outdir = ROOT / "evidence/questions" / name
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"rank{rank}.json"
        if path.exists():
            raise ValueError("question round already answered on this rank")
        body = {"model": profile["model_id"], "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}],
            "temperature": 0, "seed": profile["seed"], "max_tokens": QUESTION_MAX_TOKENS}
        rendered = c.render_prompt(body)
        inputs = contract.tokenizer.encode(rendered, add_special_tokens=False).ids
        if b"".join(contract.vocab.piece(i) for i in inputs) != rendered.encode():
            raise ValueError("tokenization does not round-trip")
        if len(inputs) + body["max_tokens"] > profile["max_model_len"]:
            raise ValueError("question too long for the context window")
        sp = SamplingParams(temperature=0, top_p=1.0, top_k=-1,
            ignore_eos=False, max_tokens=body["max_tokens"], seq_seed=body["seed"],
            eos_id=SPECIAL["eos"], first_n_token_no_eos=0)
        effective = {key: getattr(sp, key) for key in (
            "temperature", "top_p", "top_k", "ignore_eos", "max_tokens",
            "seq_seed", "eos_id", "first_n_token_no_eos")}
        save(ROOT / "evidence" / f"ready-rank{rank}.json", {
            "state": "busy", "heartbeat": time.time(), "rank": rank,
            "pid": os.getpid(), "round": name, "deadline": deadline()})
        began = time.monotonic()
        sequence = llm._add_request(None, sp, inputs, None)
        ids, finish = [], None
        last_saved = 0
        while True:
            unfinished = llm.llm_engine.has_unfinished_requests()
            outputs = llm.llm_engine.step() if unfinished else llm.llm_engine.flush_inflight_reqs()
            for native in outputs:
                output = RequestOutput.clone(native)
                if output.request_id != sequence.request_id:
                    raise RuntimeError("unexpected request id from engine")
                ids.append(int(output.token_id))
                total_tokens += 1
                if output.finished:
                    finish = output.finish_reason
            if time.monotonic() - last_saved > 5:
                save(outdir / f"progress-rank{rank}.json", {
                    "at": time.time(), "rank": rank, "output_tokens": len(ids),
                    "elapsed_seconds": time.monotonic() - began})
                last_saved = time.monotonic()
            if not unfinished:
                break
        if finish is None or not ids:
            raise RuntimeError("engine returned no finished output")
        pieces = [contract.vocab.piece(i) for i in ids]
        raw = b"".join(pieces)
        try:
            reasoning, content = channels(ids, pieces, True, finish, SPECIAL)
            channel_error = None
        except Exception as error:  # e.g. length cutoff inside a UTF-8 codepoint; raw bytes are still kept
            reasoning, content, channel_error = None, None, f"{type(error).__name__}: {error}"
        receipt = {"schema": "postfiat.governance-question-receipt.v1", "question": question,
            "base_native_profile_sha256": contract.profile_hash,
            "model_revision": profile["model_revision"], "request": body,
            "rendered_prompt_sha256": c.sha(rendered.encode()), "input_token_ids": inputs,
            "effective_sampling": effective, "output_token_ids": ids,
            "output_token_ids_sha256": c.sha(json.dumps(ids).encode()),
            "finish_reason": finish, "raw_sha256": c.sha(raw),
            "raw_output_base64": base64.b64encode(raw).decode(),
            "reasoning": reasoning, "content": content, "channel_error": channel_error,
            "execution": {"host": host, "rank": rank, "pid": os.getpid(),
                "elapsed_seconds": time.monotonic() - began,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        with path.with_suffix(".bin").open("wb") as stream:
            stream.write(raw)
        seal(path, receipt)
        total_complete += 1
        print(json.dumps({"question_complete": name, "rank": rank,
            "output_tokens": len(ids), "finish_reason": finish,
            "raw_sha256": receipt["raw_sha256"]}), flush=True)

    def run_frozen_round(command):
        nonlocal a
        if set(command) != {"round", "mode", "copies"}:
            raise ValueError("Invalid command schema")
        name = command["round"]
        if not (isinstance(name, str) and 1 <= len(name) <= 80 and all(x.isalnum() or x in "_-" for x in name)) or name == "boot-smoke":
            raise ValueError("Invalid round name")
        if command["mode"] not in ("canary", "full") or type(command["copies"]) is not int or command["copies"] not in (1, 2):
            raise ValueError("Invalid frozen-job command")
        a.round, a.mode, a.copies = name, command["mode"], command["copies"]
        rdir = ROOT / "evidence" / name
        rdir.mkdir(parents=True, exist_ok=True)
        chosen = (list(contract.rows) if a.mode == "full" else
                  [next(s for s, r in contract.rows.items() if r["context"]["ticker"] == t) for t in CANARIES])
        mine_now = [sid for i, sid in enumerate(chosen) if i % dp_size == rank]
        for sid in mine_now:
            for copy in range(a.copies):
                if receipt_path(rdir, sid, copy, "first_pass").exists():
                    raise ValueError("round already run")
        save(ROOT / "evidence" / f"ready-rank{rank}.json", {
            "state": "busy", "heartbeat": time.time(), "rank": rank,
            "pid": os.getpid(), "round": name, "deadline": deadline()})
        tasks = [(sid, copy, "", None) for sid in mine_now for copy in range(a.copies)]
        before = total_complete
        for stage in c.STAGES:
            tasks, completed = generate(tasks, stage)
        seal(rdir / f"complete-rank{rank}.json", {
            "security_ids": mine_now, "copies": a.copies, "rank": rank,
            "pid": os.getpid(), "host": host, "round": name,
            "stage_count": total_complete - before, "output_tokens": total_tokens,
            "native_profile_sha256": contract.profile_hash})

    try:
        if not a.resume:
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

        # Keep the qualified model in GPU memory and serve the command queue.
        # Nothing in this loop may end the process except the lease deadline or the stop file.
        command_root = ROOT / "commands"
        command_root.mkdir(exist_ok=True)
        errors = ROOT / "evidence/command-errors"
        errors.mkdir(parents=True, exist_ok=True)

        def finished(p):
            if (errors / f"{p.stem}-rank{rank}.json").exists():
                return True
            try:
                command = c.load(p)
            except Exception:
                return False
            if command.get("mode") == "question":
                return (ROOT / "evidence/questions" / str(command.get("round")) / f"rank{rank}.json").exists()
            return (ROOT / "evidence" / str(command.get("round")) / f"complete-rank{rank}.json").exists()

        seen = set()
        while time.time() < deadline() - 30 and not (ROOT / "stop").exists():
            save(ROOT / "evidence" / f"ready-rank{rank}.json", {
                "state": "ready", "heartbeat": time.time(), "rank": rank,
                "pid": os.getpid(), "native_profile_sha256": contract.profile_hash,
                "model_revision": profile["model_revision"],
                "startup_seconds": startup, "completed_stages": total_complete,
                "deadline": deadline(), "boot_round": "boot-smoke", "resumed": a.resume})
            pending = [p for p in sorted(command_root.glob("*.json")) if p.name not in seen and not finished(p)]
            if not pending:
                time.sleep(2)
                continue
            command_path = pending[0]
            seen.add(command_path.name)
            try:
                command = c.load(command_path)
                if not isinstance(command, dict):
                    raise ValueError("command must be a JSON object")
                if command.get("mode") == "question":
                    generate_question(command)
                else:
                    run_frozen_round(command)
            except Exception as error:
                import traceback
                save(errors / f"{command_path.stem}-rank{rank}.json", {
                    "at": time.time(), "rank": rank, "command_file": command_path.name,
                    "type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc()[-4000:]})
                print(json.dumps({"command_error": command_path.name, "rank": rank,
                                  "error": f"{type(error).__name__}: {error}"}), flush=True)
    finally:
        llm.close()


if __name__ == "__main__":
    sys.path.insert(0, "/workspace/DeepInfer")
    from examples._launch import launch_dp
    launch_dp(main, 4)
