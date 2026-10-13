# SPDX-License-Identifier: Apache-2.0
"""Validate cold/native-GPU output equality before using the routing fixtures."""

# Standard
from pathlib import Path
import argparse
import re

# Third Party
from fixtures import fixture
from run import write_json
from scenarios import Driver


def main() -> None:
    """Check both replicas with targeted cold and native GPU-only requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reuse-warmed-processes", action="store_true")
    args = parser.parse_args()
    driver = Driver(args, prepare_kernels=not args.reuse_warmed_processes)
    results = []
    try:
        for owner in ("a", "b"):
            for index in (1401, 1501, 1701):
                tokens = fixture(index, chunk_size=driver.state["chunk_size"])
                ledger = driver.ledger(owner)
                assert not ledger.prefix(tokens, "CPU") and not ledger.prefix(
                    tokens, "GPU"
                )
                cold = driver.request(tokens, owner)
                expected = (
                    len(tokens)
                    // driver.state["native_block_size"]
                    * driver.state["native_block_size"]
                )
                driver.wait(
                    lambda owner=owner, tokens=tokens, expected=expected: sum(
                        block["block_size"]
                        for block in driver.ledger(owner).prefix(tokens, "GPU")
                    )
                    == expected,
                    "independent native GPU placement before fixture control",
                )
                before = driver.snapshot(f"{owner}-{index}-before-repeat")
                repeat = driver.request(tokens, owner)
                after = driver.snapshot(f"{owner}-{index}-after-repeat")
                delta = after[owner]["mp"][len(before[owner]["mp"]) :]
                assert not any(event["event"] == "mp.retrieve.end" for event in delta)
                assert not after[owner]["reads"][len(before[owner]["reads"]) :]
                trace = after[owner]["engine_log"][len(before[owner]["engine_log"]) :]
                hits = re.findall(
                    r"MP_KV_TRACE (?:lookup|gpu_match).*?gpu_hit_tokens=(\d+)", trace
                )
                assert hits and set(map(int, hits)) == {expected}, trace
                assert cold["output"] == repeat["output"], (cold, repeat)
                results.append(
                    {
                        "owner": owner,
                        "index": index,
                        "status": "PASS",
                        "cold": cold,
                        "native_gpu": repeat,
                        "engine_trace": trace,
                    }
                )
                write_json(args.artifacts / "fixture-control.json", results)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
