import os 
import torch
from pathlib import Path
from datasets import load_dataset 
from tqdm import trange

os.environ["TORCH_CUDA_ARCH_LIST"] = "Hopper"

class CompileError(Exception):
    pass

class ExecutionError(Exception):
    pass

class NoKernelFoundError(Exception):
    pass

def fast_check(problem_id: int, device: str = "cuda:0"):
    ds = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    ds = ds.filter(lambda x: x["problem_id"] == problem_id)
    
    original_src = ds[0]["code"].strip()
    # print(original_src)

    orig_ctx = {}
    compile(original_src, "<string>", "exec")
    exec(original_src, orig_ctx)
    
    try:
      target_path = f"runs/test_hf_level_1/level_1_problem_{problem_id}_sample_0_kernel.py"
      target_src = Path(target_path).read_text().strip()
    except FileNotFoundError as e:
      raise NoKernelFoundError(f"NoKernelFoundError: {e}")

    target_ctx = {}
    compile(target_src, "<string>", "exec")
    try:
      exec(target_src, target_ctx)
    except Exception as e:
      raise ExecutionError(f"ExecutionError: {e}")

    Model = orig_ctx.get("Model")
    ModelNew = target_ctx.get("ModelNew")

    if ModelNew is None:
      raise CompileError("ModelNew is not defined")

    init_inputs = orig_ctx.get("get_init_inputs")()
    inputs = orig_ctx.get("get_inputs")()
    inputs = [x.to(device) for x in inputs]
    
    try:
      with torch.inference_mode():
          ref = Model(*init_inputs).to(device)
          cand = ModelNew(*init_inputs).to(device)
          y_ref = ref(*inputs)
          y_cand = cand(*inputs)
    except Exception as e:
      raise ExecutionError(f"ExecutionError: {e}")

    if torch.allclose(y_ref, y_cand, rtol=1e-6, atol=1e-6):
        print("PASS ✅")
    else:
        print("FAIL ❌")
        # print("ref:", y_ref.item())
        # print("cand:", y_cand.item())
        print("diff:", (y_ref - y_cand).abs().max().item())

if __name__ == "__main__":
    for problem_id in trange(52, 101):
        try:
          fast_check(problem_id, device="cuda:0")
        except CompileError as e:
          print(e)
          continue 
        except ExecutionError as e:
          print(e)
          continue
        except NoKernelFoundError as e:
          print(e)
          continue