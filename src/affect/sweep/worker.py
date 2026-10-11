"""The worker: take the next job from the queue, run it into its own folder, repeat (plan 5.1).

Started by scripts/worker.sh, which pins it to one GPU with CUDA_VISIBLE_DEVICES, so inside the
process that GPU is cuda:0. It exits when the queue is empty, when queue/STOP exists, when
queue/STOP_gpu<g> exists for its own GPU (worker ID gpu<g>_<n>; added 2026-10-03), or when the
disk has less free space than --min-free-gb. Every job ends with one line starting "JOB DONE" or
"JOB FAILED", so the logs can be counted without matching library warnings.
"""
import datetime
import json
import os
import shutil
import time
import traceback

from affect import paths
from affect.runs.store import AlreadyComplete, OutputFolder
from affect.sweep.queue import Queue


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def execute(spec: dict, device, log) -> dict:
    """Run one job into artifacts/runs/<sweep_id>/<run_id>/. Returns its validation metrics."""
    from affect.train.loop import train_job

    folder = OutputFolder(paths.runs_dir(spec["sweep_id"]), spec["run_id"]).begin()
    lines = []

    def tee(message: str) -> None:
        lines.append(message)
        log(message)

    try:
        folder.write_json("spec.json", spec)
        files = train_job(spec, device, tee)
        for name, content in files.items():
            if name.endswith(".json"):
                folder.write_json(name, content)
            else:
                folder.write_text(name, content)
        folder.write_text("log.txt", "\n".join(lines) + "\n")
        folder.complete()
    except BaseException as err:
        try:
            folder.write_text("log.txt", "\n".join(lines) + "\n")
        except FileExistsError:
            pass
        folder.fail(err)
        raise
    return files["metrics_val.json"]


def run(worker_id: str, expected_gpus: int, min_free_gb: float, max_consecutive_failures: int) -> int:
    """max_consecutive_failures: stop after this many failed jobs in a row, so a problem that fails
    every job (a broken GPU, a bug) cannot move the whole queue to failed/ overnight."""
    import torch

    from affect.env import require_gpus

    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1")
    require_gpus(expected_gpus)
    device = torch.device("cuda", 0)
    queue = Queue(paths.queue_dir())
    print(f"{_now()} worker {worker_id} on {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}), torch {torch.__version__}", flush=True)
    done = failed = skipped = in_a_row = 0
    while True:
        if queue.stop_requested(worker_id):
            print(f"{_now()} worker {worker_id}: queue/STOP or this GPU's STOP file exists, stopping", flush=True)
            break
        free_gb = shutil.disk_usage(paths.repo_root()).free / 1e9
        if free_gb < min_free_gb:
            print(f"{_now()} worker {worker_id}: {free_gb:.1f} GB free, below {min_free_gb}; stopping", flush=True)
            break
        claimed = queue.take(worker_id)
        if claimed is None:
            print(f"{_now()} worker {worker_id}: queue empty", flush=True)
            break
        spec = json.loads(claimed.read_text(encoding="utf-8"))
        started = time.perf_counter()
        print(f"{_now()} worker {worker_id}: start {spec['run_id']} ({spec['stage']}: {spec['label']})", flush=True)
        try:
            metrics = execute(spec, device, lambda m: print(m, flush=True))
        except AlreadyComplete:
            queue.finish(claimed, "done")
            skipped += 1
            print(f"{_now()} JOB SKIPPED {spec['run_id']}: already complete", flush=True)
            continue
        except Exception as err:
            queue.finish(claimed, "failed")
            failed += 1
            in_a_row += 1
            traceback.print_exc()
            print(f"{_now()} JOB FAILED {spec['run_id']}: {type(err).__name__}: {err}", flush=True)
            del err
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if in_a_row >= max_consecutive_failures:
                print(f"{_now()} worker {worker_id}: {in_a_row} jobs failed in a row, stopping so the rest of the "
                      f"queue stays pending", flush=True)
                break
            continue
        queue.finish(claimed, "done")
        done += 1
        in_a_row = 0
        print(f"{_now()} JOB DONE {spec['run_id']} best val UA {metrics['best_val_ua']:.4f} at epoch "
              f"{metrics['best_epoch']} of {metrics['epochs_run']}, {time.perf_counter() - started:.0f}s", flush=True)
        torch.cuda.empty_cache()
    print(f"{_now()} worker {worker_id} finished: {done} done, {failed} failed, {skipped} skipped", flush=True)
    return 0
