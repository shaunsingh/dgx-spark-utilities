# dgx-spark-utilities

Utilities for working with DGX Spark, mostly garbage testing but some useful bits:

- `scripts/Dockerfile.vllm`

Builds an optimized docker image for vLLM & Torch w/ support for 12.1a, optimized triton, prerelease flashinfer, BLAS, etc.

The default nvidia release is missing a few critical fixes from master

`docker build -f scripts/Dockerfile.vllm -t vllm:intellect3 .`

- `vllm.sh`, `sglang.sh`, `tensorrt.sh`, `model_catalog.sh`

Shims for testing models against nvidia's containers

- `cache_model.sh`

Caches model to huggingface model directory w/ progress bar. Tired of no progressbar w/ VLLM

- `models/`, `broken/`, `rec/`

Scripts to run aforementinoned models w/ custom config, `broken` are untested, `rec` are recommended

- `scripts/nvfp4i3.sh` 

Example of quantizing FP8 to NVFP4

- `scripts/proton.sh` 

Build ARM64+Fex proton 

- `benchmark.sh`, `results.txt`

Benchmarking aforementioned runtimes, scripts, models

- `sync_to_spark.sh`

syncs directory back and forth w/ spark

- `nvidiawebui.sh`

opens webui

