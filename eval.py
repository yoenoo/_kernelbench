import torch
import contextlib

# --- Reference & Candidate --- #
# Assumes you have Model, ModelNew, get_inputs(), get_init_inputs() defined.

def _move_to(x, device):
    if isinstance(x, (list, tuple)):
        return [t.to(device, non_blocking=True) for t in x]
    return x.to(device, non_blocking=True)

def _check_same(x, y, rtol=1e-6, atol=1e-6):
    if x.shape != y.shape:
        return False, f"shape mismatch: {x.shape} vs {y.shape}"
    if x.dtype != y.dtype:
        return False, f"dtype mismatch: {x.dtype} vs {y.dtype}"
    ok = torch.allclose(x, y, rtol=rtol, atol=atol)
    if not ok:
        max_abs_err = (x - y).abs().max().item()
        return False, f"allclose failed (rtol={rtol}, atol={atol}), max|Δ|={max_abs_err:.3e}"
    return True, "ok"

def fast_correctness_check(
    reference_cls,
    candidate_cls,
    get_inputs_fn,
    get_init_inputs_fn,
    device=None,
    rtol=1e-6,
    atol=1e-6,
    use_small_inputs=True,
):
    # 1) Device + deterministic-ish run
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    # 2) Instantiate models
    ref = reference_cls().to(device)
    cand = candidate_cls().to(device)

    # 3) (Optional) tiny inputs to be super fast, overrides get_inputs() shape
    #    We’ll run two quick cases: random + a couple edgey values
    tiny_cases = []
    if use_small_inputs:
        # A very small tensor (fast) with hinge-loss-friendly targets
        bs, n = 8, 16
        preds = torch.randn(bs, n)
        targets = (torch.randint(0, 2, (bs, n)).float() * 2 - 1)
        tiny_cases.append([preds, targets])

        # Edge case: zeros and large magnitudes
        preds2 = torch.cat([torch.zeros(bs, n//2), torch.full((bs, n - n//2), 10.0)], dim=1)
        targets2 = torch.cat([torch.full((bs, n//2), -1.0), torch.ones(bs, n - n//2)], dim=1)
        tiny_cases.append([preds2, targets2])

    # 4) Real inputs (single batch) if you want to also test the given generator
    real_inputs = get_inputs_fn()

    # 5) Move & warm up once
    @contextlib.contextmanager
    def _inference():
        with torch.inference_mode():
            yield

    def _run_once(model, inputs):
        out = model(*inputs)
        # Ensure scalar (hinge loss returns mean scalar)
        if out.ndim != 0:
            out = out.mean()
        return out

    # If your model needs init_inputs() (e.g., kernels), call it once
    init_inps = get_init_inputs_fn()
    if isinstance(init_inps, (list, tuple)) and len(init_inps) > 0:
        _ = [t.to(device) for t in init_inps]  # typically a no-op for your example

    def _do_case(inputs_cpu, label):
        inputs = _move_to(inputs_cpu, device)
        # warm-up (esp. for CUDA kernels)
        with _inference():
            _ = _run_once(ref, inputs)
            _ = _run_once(cand, inputs)
            if device == "cuda":
                torch.cuda.synchronize()

        # measured (optional) — here just the correctness
        with _inference():
            y_ref = _run_once(ref, inputs)
            y_cand = _run_once(cand, inputs)
            if device == "cuda":
                torch.cuda.synchronize()

        ok, msg = _check_same(y_ref, y_cand, rtol=rtol, atol=atol)
        status = "PASS" if ok else "FAIL"
        # Also report max abs err for transparency
        max_abs_err = (y_ref - y_cand).abs().max().item()
        print(f"[{label}] {status}: ref={y_ref.item():.6f}, cand={y_cand.item():.6f}, max|Δ|={max_abs_err:.3e} ({msg})")
        return ok

    all_ok = True
    # Tiny cases: fastest signal
    for i, case in enumerate(tiny_cases):
        all_ok &= _do_case(case, label=f"tiny_case_{i+1}")

    # Single real case (if you want to include one run of your generator)
    all_ok &= _do_case(real_inputs, label="real_inputs")

    return all_ok

# ---- Run it ----
if __name__ == "__main__":
    ok = fast_correctness_check(
        reference_cls=Model,
        candidate_cls=ModelNew,
        get_inputs_fn=get_inputs,
        get_init_inputs_fn=get_init_inputs,
        rtol=1e-6,
        atol=1e-6,
        use_small_inputs=True,   # turn this off if you really want the huge batch
    )
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
