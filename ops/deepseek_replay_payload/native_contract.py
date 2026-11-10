"""Native DeepInfer receipts bound to the existing frozen scoring contract."""
import base64
import json
from pathlib import Path

from tokenizers import Tokenizer
from replay_contract import core as c
from replay_contract.runtime import channels
from replay_contract.verify_job import Vocabulary
from replay_contract.job import verify_source

ROOT = Path(__file__).resolve().parent
SPECIAL = {"bos": 0, "eos": 1, "think_end": 128822}
CANARIES = ["ENPH", "INTC", "INRO", "NVT"]


class Contract:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.job = c.load(self.root / "job.json")
        self.profile = c.load(self.root / "native-profile.json")
        self.profile_hash = c.digest(self.profile)
        self.vocab = Vocabulary(self.root / "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.rows = {x["security_id"]: x for x in self.job["securities"]}
        self.inputs = c.load(self.root / "prepared-inputs.json")["requests"]
        assert len(self.rows) == len(self.inputs) == 452
        assert self.job["scoring_profile"] == c.P
        assert c.digest(self.job) == self.profile["job_sha256"]
        verify_source(self.job["source"], (self.root / "source.json").read_bytes())
        for name, digest in self.profile["source_sha256"].items():
            assert c.sha((self.root / name).read_bytes()) == digest, "Runner source drift"

    def request(self, security_id, stage, notes=""):
        context = self.rows[security_id]["context"]
        body = c.request(context, stage, notes)
        prompt = c.render_prompt(body)
        ids = (self.inputs[security_id]["input_token_ids"] if stage != "capped_finalizer"
               else self.tokenizer.encode(prompt, add_special_tokens=False).ids)
        if stage == "first_pass":
            assert c.digest(body) == self.rows[security_id]["first_pass_request_sha256"]
        assert b"".join(self.vocab.piece(i) for i in ids) == prompt.encode()
        assert len(ids) + body["max_tokens"] <= self.profile["max_model_len"], "Full stage budget exceeds context"
        so_args = {"schema_type": "json", "schema": json.dumps({"type": "object"})}
        if stage != "capped_finalizer":
            so_args["trigger_token_ids"] = [SPECIAL["think_end"]]
        effective = {"temperature": 1e-5, "top_p": 1.0, "top_k": -1,
                     "ignore_eos": False, "max_tokens": body["max_tokens"],
                     "seq_seed": body["seed"], "eos_id": SPECIAL["eos"],
                     "first_n_token_no_eos": 0, "so_args": so_args}
        return {"request": body, "request_sha256": c.digest(body),
                "rendered_prompt_sha256": c.sha(prompt.encode()), "input_token_ids": ids,
                "effective_sampling": effective}

    def observe(self, sid, stage, notes, parent, token_ids, finish_reason, metadata):
        request = self.request(sid, stage, notes)
        assert token_ids and len(token_ids) <= request["request"]["max_tokens"]
        if finish_reason == "length":
            assert len(token_ids) == request["request"]["max_tokens"]
        pieces = [self.vocab.piece(i) for i in token_ids]
        raw = b"".join(pieces)
        reasoning, content = channels(token_ids, pieces, stage != "capped_finalizer", finish_reason, SPECIAL)
        decision = c.transition(stage, finish_reason, reasoning, content)
        value = {"schema": "corbanu.deepinfer-stage.v1", "security_id": sid, "stage": stage,
                 "native_profile_sha256": self.profile_hash, "parent_output_sha256": parent,
                 **request, "output_token_ids": token_ids, "finish_reason": finish_reason,
                 "raw_sha256": c.sha(raw), "raw_output_base64": base64.b64encode(raw).decode(),
                 "reasoning": reasoning, "content": content, "transition": decision,
                 "execution": metadata, "publication_allowed": False}
        value["commitment"] = commitment(value)
        return value, raw

    def verify(self, value, sid, stage, notes, parent):
        expected, raw = self.observe(sid, stage, notes, parent, value["output_token_ids"],
                                     value["finish_reason"], value["execution"])
        assert expected == value, "Receipt binding or raw/token mismatch"
        if value["finish_reason"] == "stop":
            assert self.tokenizer.decode(value["output_token_ids"], skip_special_tokens=False).encode() == raw
        return value["transition"]


def commitment(value):
    keys = ("security_id", "stage", "native_profile_sha256", "parent_output_sha256",
            "request_sha256", "rendered_prompt_sha256", "input_token_ids", "output_token_ids",
            "finish_reason", "raw_sha256", "reasoning", "content")
    return c.digest({k: value[k] for k in keys})


def receipt_path(directory, sid, copy, stage):
    return Path(directory) / "securities" / c.sha(sid.encode()) / str(copy) / (stage + ".json")


def verify_group(contract, directory, ids, copies):
    directory = Path(directory)
    result = []
    expected_paths = set()
    for sid in ids:
        for copy in range(copies):
            stage, notes, parent = "first_pass", "", None
            chain = []
            while stage:
                path = receipt_path(directory, sid, copy, stage)
                expected_paths.add(path)
                value = c.load(path)
                decision = contract.verify(value, sid, stage, notes, parent)
                chain.append({"stage": stage, "commitment": value["commitment"],
                              "raw_sha256": value["raw_sha256"],
                              "tokens": len(value["output_token_ids"])})
                parent = value["commitment"]
                if decision["action"] == "accepted":
                    result.append({"security_id": sid, "copy": copy, "stages": chain, **decision["parsed"]})
                    stage = None
                elif decision["action"] == "next":
                    stage, notes = decision["stage"], decision["notes"]
                else:
                    raise ValueError("Stage policy failed: " + sid)
    actual = set((directory / "securities").glob("*/*/*.json"))
    assert actual == expected_paths, "Missing, extra, or duplicate stage coverage"
    return result
