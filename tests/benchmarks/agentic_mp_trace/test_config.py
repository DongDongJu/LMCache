# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.agentic_mp_trace.config import expand_env, load_yaml_config


def test_config_catalog_contains_required_datasets():
    config = load_yaml_config("benchmarks/agentic_mp_trace/config.example.yaml")
    catalog = config["dataset_catalog"]
    for name in [
        "tau_bench_current",
        "swe_bench_lite",
        "webarena",
        "mind2web",
        "appworld",
        "toolbench",
        "agentbench",
        "browsergym",
        "workarena",
        "the_agent_company",
        "gaia",
        "osworld",
    ]:
        assert name in catalog
        assert catalog[name]["adapter"]


def test_environment_expansion(monkeypatch):
    monkeypatch.setenv("MODEL_SMALL", "Qwen/Qwen3-0.6B")
    payload = {"model": "${MODEL_SMALL}", "missing": "${NO_SUCH_MODEL}"}
    expanded = expand_env(payload)
    assert expanded["model"] == "Qwen/Qwen3-0.6B"
    assert expanded["missing"] == "${NO_SUCH_MODEL}"

