set -xe

# 1. Generate responses and store kernels locally to runs/{run_name} directory
# python3 scripts/generate_samples.py \
#   run_name=test_hf_level_1 \
#   dataset_src=huggingface \
#   level=1 \
#   num_workers=50 \
#   server_type=sglang \
#   model_name=Qwen/Qwen3-14B \
#   temperature=0

# 2. Evaluate on all generated kernels in runs/{run_name} directory
python3 scripts/eval_from_generations.py \
  run_name=test_hf_level_1 \
  dataset_src=local \
  level=1 \
  num_gpu_devices=1 \
  timeout=300 \
  build_cache=True \
  num_cpu_workers=50

# # If you like to speedup evaluation, you can use parallelize compilation on CPUs before getting to evluation on GPUs
# # add build_cache=True and num_cpu_workers=<num_cpu_workers> to the command