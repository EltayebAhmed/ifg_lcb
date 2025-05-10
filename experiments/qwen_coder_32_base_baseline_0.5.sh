#!/bin/bash
python3.11 -m lcb_runner.runner.main --model generic-vanilla-gllm --scenario codegeneration --evaluate --start_date 2025-01-01 --end_date 2025-05-01 --multiprocess=12 --cache_batch_size=2 --use_cache --server-address=http://3.87.42.244:8181 --n=10 --temperature=0.5
