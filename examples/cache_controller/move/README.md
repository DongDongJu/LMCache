# LMCache Move/Migrate
This is an example to demonstrate how to move/migrate a request's KV cache across LMCacheEngines externally.

## Prerequisites
Your server should have at least 2 GPUs. [NIXL](https://github.com/ai-dynamo/nixl) is required to be installed.

This will use port 8000 and 8001 for 2 vllms and port 8500 and 8501 for the corresponding LMCache workers. Also, ports 8200, 8201, 8202 and 8203 are used for p2p KV cache transfer. The controller itself occupies port 9000, 8300 and 9400.

## Steps
1. Start two vllm engines at port 8000 and port 8001:

```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=instance1.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

```bash
PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=instance2.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

2. Start the lmcache controller at port 9000 and the monitor at port 9001:

```bash
PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
```

3. Use the GDPVal dataset to drive a request. First make sure the dataset has been
   downloaded to `/xfs1/alex/dataset/openai_gdpval`. Then run the helper script:

```bash
python use_gdpval_prompt.py \
  --dataset-dir /xfs1/alex/dataset/openai_gdpval \
  --index 0 \
  --max-tokens 32
```

This sends the selected dataset prompt to the vLLM server on port `8000`, prints the
completion text, and returns the token ids needed in the next step. You can pass
`--task-id <uuid>` instead of `--index` to target a specific task and use
`--save-token-file` to persist the ids for later reuse.
Install the `datasets` and `requests` packages if they are not already available.

4. Move the request's KV cache from vllm engine 1's CPU to vllm engine 2's CPU
   using the token ids reported by the helper script:
```bash
curl -X POST http://localhost:9000/move \
  -H "Content-Type: application/json" \
  -d '{
    "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
    "new_position": ["lmcache_instance_2", "LocalCPUBackend"],
    "tokens": [/* paste the array emitted by use_gdpval_prompt.py */]
  }'
```
You should be able to see a return message indicating the KV cache has started to be moved in the system:

```plaintext
{"num_tokens": <token_count>, "event_id": "xxx"}
```

`num_tokens` reports how many tokens are tracked for the request in the system. The returned `event_id` can be used to check the status of the move operation (this functionality is coming soon).
