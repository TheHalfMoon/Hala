"""Hala H86 — Kaggle private personal non-commercial CUDA probe only.
No Hala model weights, private datasets, or commercial training."""
import json
import os
import platform
from pathlib import Path
import torch
out = Path("/kaggle/working")
out.mkdir(parents=True, exist_ok=True)
data = dict(schema="hala.h86.private-kaggle-gpu-research-probe.v1",
            scope="personal_noncommercial_research_toy_optimizer_only",
            hala_model_training=False, dataset_accessed=False,
            commercial_training_qualified=False, paid_compute_requested=False,
            kaggle_session=bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")),
            cuda_available=bool(torch.cuda.is_available()),
            gpu_count=torch.cuda.device_count(),
            torch_version=torch.__version__)
if data["kaggle_session"] and data["cuda_available"]:
    data["gpu_names"] = [torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count())]
    data["gpu_vram_bytes"] = [int(torch.cuda.get_device_properties(i).total_memory)
                              for i in range(torch.cuda.device_count())]
    param = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [2.0, -2.0]], device="cuda:0"))
    opt = torch.optim.SGD([param], lr=0.01)
    old = param.detach().clone()
    opt.zero_grad()
    loss = param.square().mean()
    loss.backward()
    data["toy_gradient_norm"] = float(param.grad.norm().item())
    opt.step()
    torch.cuda.synchronize(0)
    data["toy_step_updated"] = bool(not torch.equal(old, param.detach()))
    data["status"] = "PASS_GPU_TOY_STEP_ONLY" if (
        data["toy_step_updated"] and data["toy_gradient_norm"] > 0
    ) else "FAILED_GPU_TOY_STEP"
else:
    data["status"] = "NO_ALLOCATED_KAGGLE_GPU"
(out / "hala-h86-private-gpu-probe.json").write_text(
    json.dumps(data, sort_keys=True, indent=2), encoding="utf8")
print("HALA_H86_GPU_RESEARCH_PROBE", json.dumps(data, sort_keys=True), flush=True)
if data["status"] != "PASS_GPU_TOY_STEP_ONLY":
    raise SystemExit("No qualifying synthetic optimizer evidence")
