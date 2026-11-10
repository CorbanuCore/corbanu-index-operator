"""Verify a retained host run without a GPU or access to another host's outputs."""
import argparse
from pathlib import Path
from evidence import save
from native_contract import CANARIES, Contract, receipt_path, verify_group
from replay_contract import core as c


def verify_canary(root):
    contract = Contract()
    ids = [next(s for s, r in contract.rows.items() if r["context"]["ticker"] == t) for t in CANARIES]
    groups = []
    summaries = []
    for name, copies in (("canary-cold", 1), ("canary-fresh", 2)):
        directory = root / name
        group = verify_group(contract, directory, ids, copies)
        groups.append(group)
        for rank in range(4):
            summaries.append(c.load(directory / f"complete-rank{rank}.json"))
        for sid in ids:
            first = c.load(receipt_path(directory, sid, 0, "first_pass"))
            # Same exact notes generation as the explicitly labelled diagnostic.
            item = next(x for x in group if x["security_id"] == sid and x["copy"] == 0)
            last = c.load(receipt_path(directory, sid, 0, item["stages"][-1]["stage"]))
            notes = (last["reasoning"] + last["content"]) * 40
            rank = first["execution"]["rank"]
            for copy in range(copies):
                stress = c.load(directory / "stress" / f"rank{rank}-copy{copy}.json")
                assert stress["execution"]["diagnostic_only"] is True
                decision = contract.verify(stress, sid, "capped_finalizer", notes, None)
                assert decision["action"] == "accepted"
                baseline = c.load(root / "canary-cold/stress" / f"rank{rank}-copy0.json")
                assert stress["commitment"] == baseline["commitment"], "Finalizer diagnostic mismatch"
    baseline = {x["security_id"]: {k: v for k, v in x.items() if k != "copy"} for x in groups[0]}
    for x in groups[1]:
        assert {k: v for k, v in x.items() if k != "copy"} == baseline[x["security_id"]], "Fresh-process or batch mismatch"
    for rank in range(4):
        assert len({x["pid"] for x in summaries if x["rank"] == rank}) == 2
    result = {"status": "four_name_fresh_process_and_batch_passed", "security_ids": ids,
              "accepted_responses": 12, "finalizer_diagnostic_responses": 12,
              "native_profile_sha256": contract.profile_hash, "scores": list(baseline.values()),
              "completed": summaries, "publication_allowed": False}
    save(root / "canary-comparison.json", result)
    return result


def verify_full(root):
    contract = Contract()
    ids = list(contract.rows)
    result = verify_group(contract, root / "full", ids, 1)
    assert len(result) == 452
    completed = [c.load(root / "full" / f"complete-rank{rank}.json") for rank in range(4)]
    assert sorted(s for x in completed for s in x["security_ids"]) == sorted(ids)
    scores = [{"security_id": x["security_id"], "score": x["score"], "brief": x["brief"]} for x in result]
    save(root / "scores.json", scores)
    report = {"status": "complete_host_index_recorded", "security_count": len(result),
              "stage_count": sum(len(x["stages"]) for x in result),
              "native_profile_sha256": contract.profile_hash, "scores_sha256": c.digest(scores),
              "rows": result, "completed": completed, "publication_allowed": False}
    save(root / "full-verification.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["canary", "full"])
    p.add_argument("root", type=Path)
    a = p.parse_args()
    report = (verify_canary if a.mode == "canary" else verify_full)(a.root)
    print(report["status"], flush=True)
