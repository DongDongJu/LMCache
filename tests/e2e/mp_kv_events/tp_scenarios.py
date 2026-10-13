# SPDX-License-Identifier: Apache-2.0
"""T07: real TP rank delay, failed reservation, shard eviction and MP load."""

# Standard
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import subprocess
import time
import urllib.error

# Third Party
from run import write_json
from scenarios import Driver, fixture


def control(driver: Driver, owner: str, **payload: object) -> dict:
    """Use the existing private MP request transport for test-only controls."""
    scheme = "grpc" if driver.state["transport"] == "grpc" else "tcp"
    script = (
        "import subprocess,sys; "
        "subprocess.run(['python3','/opt/mp-kv-events/mp_test_hooks.py',"
        "'--url',sys.argv[1],sys.argv[2]],check=True)"
    )
    raw = driver.exec(owner, script, f"{scheme}://127.0.0.1:5555", json.dumps(payload))
    # Runtime initialization may log before the single JSON control response.
    return json.loads(raw.splitlines()[-1])


def native_records(driver: Driver, owner: str, cursor: int) -> list[dict]:
    """Read the module's real immutable native records through its public API."""
    batch = control(
        driver,
        owner,
        action="events",
        cursor=cursor,
        model_name=driver.model_name(owner),
        world_size=2,
    )
    assert batch["enabled"] and not batch["lost"], batch
    return batch["events"]


def fixture_object_keys(driver: Driver, owner: str, tokens: list[int]) -> list[dict]:
    """Resolve committed keys by exact MP_TOKENS-to-L1 EventBus evidence."""
    records = driver.records(owner, "mp-events.jsonl")
    chunks: set[str] = set()
    chunk_size = driver.state["chunk_size"]
    for event in records:
        if event["event"] != "mp.tokens":
            continue
        metadata = event["metadata"]
        for token_chunk, chunk_hash in zip(
            metadata.get("token_chunks", []),
            metadata.get("chunk_hashes", []),
            strict=True,
        ):
            if token_chunk == tokens[:chunk_size]:
                chunks.add(chunk_hash["bytes_base64"])
    keys = {}
    for event in records:
        if event["event"] not in (
            "l1.write.finished",
            "l1.write_finished_and_read_reserved",
        ):
            continue
        for key in event["metadata"]["keys"]:
            content = key["chunk_hash"]["bytes_base64"]
            if content in chunks:
                encoded = {
                    "chunk_hash_hex": base64.b64decode(content).hex(),
                    "model_name": key["model_name"],
                    "kv_rank": key["kv_rank"],
                    "object_group_id": key["object_group_id"],
                    "cache_salt": key["cache_salt"],
                }
                keys[(content, key["kv_rank"], key["object_group_id"])] = encoded
    return list(keys.values())


def require_native_publication_once(
    records: list[dict], target_hashes: set[str], kind: str
) -> list[dict]:
    """Count target identities in captured native events before deduplication."""
    assert target_hashes, "No target native identities supplied"
    observed: Counter[str] = Counter()
    evidence = []
    for offset, record in enumerate(records):
        for event in record["payload"][1]:
            if event["type"] != kind or event.get("medium") not in (
                "CPU",
                "CPU_PINNED",
            ):
                continue
            matching = [
                value for value in event["block_hashes"] if str(value) in target_hashes
            ]
            if not matching:
                continue
            observed.update(str(value) for value in matching)
            evidence.append(
                {
                    "record_offset": offset,
                    "received_ns": record["received_ns"],
                    "publisher_timestamp": record["payload"][0],
                    "type": kind,
                    "medium": event["medium"],
                    "target_hashes": matching,
                }
            )
    assert dict(observed) == dict.fromkeys(target_hashes, 1), (
        f"Each target must have exactly one native CPU {kind}",
        dict(observed),
        target_hashes,
    )
    return evidence


def run_owner(driver: Driver, owner: str, index: int) -> None:
    """Prove partial/complete/removal transitions using real rank operations."""
    tokens = fixture(index, 1, driver.state["chunk_size"])
    expected_tokens = tokens[:-1]
    before = control(driver, owner, action="status")
    assert before["kv_events"]["subscribers"] == 1, before
    cursor = before["kv_events"]["next_seq"] - 1
    native_cursor = len(driver.records(owner, "native-events.jsonl"))
    control(driver, owner, action="delay_rank", rank=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(driver.request, tokens, owner)
        try:
            driver.wait(
                lambda: control(driver, owner, action="status")["kv_events_e2e"][
                    "held_stores"
                ]
                > 0,
                "one real rank STORE held",
                timeout=30,
            )
            driver.wait(
                lambda: bool(fixture_object_keys(driver, owner, tokens)),
                "other rank actually committed",
                timeout=30,
            )
            partial_keys = fixture_object_keys(driver, owner, tokens)
            assert {key["kv_rank"] & 0xFF for key in partial_keys} == {0}, partial_keys
            partial = native_records(driver, owner, cursor)
            assert not [
                event
                for event in partial
                if event["kind"] == "stored" and event["token_ids"] == expected_tokens
            ], partial
            assert not driver.ledger(owner).prefix(tokens, "CPU")
            driver.snapshot(f"T07-{owner}-partial")
        finally:
            control(driver, owner, action="release_rank")
        baseline = running.result(timeout=90)
    driver.wait(
        lambda: any(
            event["kind"] == "stored" and event["token_ids"] == expected_tokens
            for event in native_records(driver, owner, cursor)
        ),
        "all required TP objects committed",
        timeout=30,
    )
    committed = native_records(driver, owner, cursor)
    stored = [
        event
        for event in committed
        if event["kind"] == "stored" and event["token_ids"] == expected_tokens
    ]
    assert len(stored) == 1, stored
    target_hashes = {str(value) for value in stored[0]["block_hashes"]}
    driver.wait_received(owner)
    driver.active_barrier(owner)
    driver.wait(
        lambda: bool(driver.ledger(owner).prefix(tokens, "CPU")), "TP CPU publication"
    )
    native_after_release = driver.records(owner, "native-events.jsonl")[native_cursor:]
    write_json(
        driver.root / f"T07-{owner}-native-after-release.json", native_after_release
    )
    stored_publications = require_native_publication_once(
        native_after_release, target_hashes, "BlockStored"
    )
    keys = fixture_object_keys(driver, owner, tokens)
    assert {key["kv_rank"] & 0xFF for key in keys} == {0, 1}, keys
    shard = next(key for key in keys if key["kv_rank"] & 0xFF == 1)
    deletion = control(driver, owner, action="evict", keys=[shard])
    assert deletion == {"deleted": 1, "skipped": 0}, deletion
    driver.wait(
        lambda: any(
            event["kind"] == "removed"
            and event["block_hashes"] == stored[0]["block_hashes"]
            for event in native_records(driver, owner, cursor)
        ),
        "one required shard withdraws entire CPU chunk",
        timeout=30,
    )
    driver.wait_received(owner)
    driver.active_barrier(owner)
    driver.wait(
        lambda: not driver.ledger(owner).prefix(tokens, "CPU"), "TP scoped CPU removal"
    )
    native_after_removal = driver.records(owner, "native-events.jsonl")[native_cursor:]
    write_json(
        driver.root / f"T07-{owner}-native-after-removal.json", native_after_removal
    )
    # Include late publications in both counts; final ledger absence alone would
    # conceal duplicate stores followed by a valid removal.
    stored_publications = require_native_publication_once(
        native_after_removal, target_hashes, "BlockStored"
    )
    removed_publications = require_native_publication_once(
        native_after_removal, target_hashes, "BlockRemoved"
    )
    removed = [
        event
        for event in native_records(driver, owner, cursor)
        if event["kind"] == "removed"
        and event["block_hashes"] == stored[0]["block_hashes"]
    ]
    assert len(removed) == 1, removed
    assert driver.ledger(owner).prefix(tokens, "GPU"), (
        "CPU shard eviction erased GPU placement"
    )

    failed_tokens = fixture(index + 10, 1, driver.state["chunk_size"])
    status = control(driver, owner, action="fail_rank", rank=1)
    failed_cursor = status["kv_events"]["next_seq"] - 1
    failures_before = status["kv_events_e2e"]["reservation_failures"]
    driver.request(failed_tokens, owner)
    driver.wait(
        lambda: control(driver, owner, action="status")["kv_events_e2e"][
            "reservation_failures"
        ]
        == failures_before + 1,
        "one real reservation failed",
        timeout=30,
    )
    driver.wait_received(owner)
    driver.active_barrier(owner)
    failed_records = native_records(driver, owner, failed_cursor)
    assert not [
        event
        for event in failed_records
        if event["kind"] == "stored" and event["token_ids"] == failed_tokens[:-1]
    ], failed_records
    assert not driver.ledger(owner).prefix(failed_tokens, "CPU")
    # A fresh successful prefix proves normal routing/load after the fault.
    fresh = fixture(index + 20, 1, driver.state["chunk_size"])
    fresh_baseline = driver.warm(fresh, owner)
    driver.evict_gpu(fresh, owner)
    driver.probe_owner(fresh, owner)
    driver.measured(fresh, owner, True, fresh_baseline, f"T07-{owner}-recovery")
    final = control(driver, owner, action="status")
    assert final["kv_events"]["subscribers"] == 1, final
    write_json(
        driver.root / f"T07-{owner}-rank-evidence.json",
        {
            "partial_keys": partial_keys,
            "complete_keys": keys,
            "stored": stored,
            "removed": removed,
            "native_cursor_before_request": native_cursor,
            "target_native_hashes": sorted(target_hashes),
            "native_stored_publications": stored_publications,
            "native_removed_publications": removed_publications,
            "failed_records": failed_records,
            "baseline_request": baseline,
            "final_status": final,
        },
    )


def run_tp(driver: Driver) -> list[dict]:
    """Execute T07 in both owner directions for the four-GPU TP2 profile."""
    if driver.state["tp"] != 2:
        return []
    results = []
    for index, owner in enumerate(("a", "b")):
        started = time.monotonic()
        row: dict[str, object] = {"scenario": "T07", "owner": owner}
        try:
            run_owner(driver, owner, 3000 + index * 100)
            row["status"] = "PASS"
        except (
            AssertionError,
            RuntimeError,
            subprocess.CalledProcessError,
            urllib.error.URLError,
            TimeoutError,
        ) as error:
            row.update(status="FAIL", reason=str(error))
        row["duration_seconds"] = time.monotonic() - started
        results.append(row)
    return results
