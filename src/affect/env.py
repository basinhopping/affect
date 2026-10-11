"""Environment checks for the node (plan 8.3).

`environment_report` needs no GPU and runs anywhere. `gpu_report` imports torch (after this
package's MPS fix, which `import affect` has already applied), confirms CUDA sees the expected
GPUs, and on every GPU runs one training step and compares a matrix product with the CPU.
`require_gpus` is what training calls: it refuses to run unless every expected GPU is visible,
so a missed fix fails loudly instead of falling back to the CPU.
"""
import os
import platform
import shutil
import subprocess
import time
import urllib.request

from affect import paths
from affect.runs.provenance import collect


def _nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False}
    query = "index,name,memory.total,memory.used,driver_version,compute_mode"
    done = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=60)
    procs = subprocess.run([exe, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
    return {"available": True, "returncode": done.returncode, "gpus": done.stdout.strip().splitlines(),
            "stderr": done.stderr.strip()[-2000:], "compute_apps": procs.stdout.strip().splitlines()}


def _memory() -> dict:
    out = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = value.strip()
    except OSError:
        pass
    return out


def _reach(url: str) -> dict:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=20) as response:
            return {"url": url, "status": response.status}
    except Exception as err:  # the report records any failure as text
        return {"url": url, "error": f"{type(err).__name__}: {err}"}


def environment_report() -> dict:
    root = paths.repo_root()
    usage = shutil.disk_usage(root)
    return {
        "provenance": collect("affect.env.environment_report"),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory": _memory(),
        "disk_at_repo": {"path": str(root), "total_gb": round(usage.total / 1e9, 1),
                         "free_gb": round(usage.free / 1e9, 1)},
        "nvidia_smi": _nvidia_smi(),
        "network": [_reach("https://pypi.org/simple/"), _reach("https://huggingface.co")],
    }


def gpu_report(expected_gpus: int) -> dict:
    import torch

    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "expected_gpus": expected_gpus,
        "devices": [],
    }
    if not report["cuda_available"]:
        report["ok"] = False
        report["reason"] = "CUDA is not available after the MPS fix"
        return report
    for index in range(report["device_count"]):
        device = torch.device("cuda", index)
        entry = {"index": index, "name": torch.cuda.get_device_name(index)}
        try:
            free, total = torch.cuda.mem_get_info(index)
            entry.update({"free_mib": free // 2**20, "total_mib": total // 2**20})
            gen = torch.Generator().manual_seed(index)
            a = torch.randn(512, 512, generator=gen)
            b = torch.randn(512, 512, generator=gen)
            cpu = a @ b
            gpu = (a.to(device) @ b.to(device)).cpu()
            # Relative to the largest entry, so the check holds whether or not TF32 is on.
            entry["matmul_rel_diff_vs_cpu"] = float((cpu - gpu).abs().max() / cpu.abs().max())
            model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.randn(256, 64, generator=gen).to(device)
            y = torch.randint(0, 4, (256,), generator=gen).to(device)
            before = [p.detach().clone() for p in model.parameters()]
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            entry["train_step_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            entry["loss_finite"] = bool(torch.isfinite(loss).item())
            entry["parameters_changed"] = any(not torch.equal(p0, p1.detach()) for p0, p1 in zip(before, model.parameters()))
            entry["ok"] = bool(entry["loss_finite"] and entry["parameters_changed"]
                               and entry["matmul_rel_diff_vs_cpu"] < 5e-3)
        except Exception as err:  # recorded, and the check fails
            entry["ok"] = False
            entry["error"] = f"{type(err).__name__}: {err}"
        report["devices"].append(entry)
    report["ok"] = bool(report["device_count"] == expected_gpus and report["devices"]
                        and all(d["ok"] for d in report["devices"]))
    return report


def require_gpus(expected_gpus: int) -> None:
    """Refuse to continue unless CUDA sees exactly the expected number of GPUs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. On the node this usually means torch was imported "
                           "before the MPS fix. Start Python through scripts/*.sh.")
    if torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(f"expected {expected_gpus} GPUs, CUDA sees {torch.cuda.device_count()}")
