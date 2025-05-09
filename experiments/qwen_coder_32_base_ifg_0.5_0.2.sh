#!/bin/bash
python3.11 -m lcb_runner.runner.main --model generic-ifg-model --scenario codegeneration --evaluate --start_date 2024-10-01 --end_date 2025-02-01 --multiprocess=12 --cache_batch_size=16 --use_cache --server-address=http://13.218.21.212:8181 --n=10
