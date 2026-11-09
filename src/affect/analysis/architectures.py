"""The exact architecture of each model the project reports or screens, built by the training code from recorded
settings (Dr. Luitel, 2026-10-21: "I need the exact architectures"; workstream H of the week plan; added 2026-10-26).
Node only (it needs torch), on the CPU, in a few seconds:

    cd ~/affect && CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/venvs/affect/bin/python -m affect.analysis.architectures

Each model's settings come from a sweep file in sweeps/, expanded by affect.sweep.design.expand: a record's settings
are the configuration of its sweep with its configuration hash (checked in full), a screen's base is the
configuration of the sweep's base stage. The network is built exactly as affect.train.loop builds it
(MODELS.get(name).target(settings, input width, number of classes)) after torch.manual_seed(0), then put in eval mode,
and for each model this writes:
  - PyTorch's printed module tree (every layer with its sizes, kernels, strides and options);
  - the parameters of each top-level part and in all, beside n_parameters of the runs' efficiency.json when a run
    folder of the configuration is on this machine (a check that the network built here is the one trained);
  - every leaf layer's input and output shape for one utterance of 3 s, from forward hooks on the first call (a LIF
    layer runs once per time step, so the number of calls is given too): a log-mel model reads the log-mel image its
    own features compute from 48,000 samples of noise at 16 kHz, a frames head 150 frames (3 s at 50 frames a
    second) of noise of its encoder's width.
Nothing is trained and nothing is written outside artifacts/architectures/architectures_<utc>/ (architectures.txt,
architectures.json). A model that fails is reported with its error and the others still run; the exit code is 1 then.
"""
import csv
import datetime
import json
import sys
import traceback
from pathlib import Path

from affect import catalog, paths

# (name in the documents, sweep file, configuration hash) of each record: the five finalists and the baseline CNN, as
# the selection records in artifacts/selections name them (made in the Cowork container; the node has the sweep files)
RECORDS = (
    ("cnn_base_session_out (the baseline CNN)", "cnn_confirm_1",
     "cf0495a296e92ced1ea9122472b8a9d09cd6852b821b785e4798c4ee52de52b3"),
    ("cnn_final_3 (the tuned 4-block CNN)", "cnn_confirm_3",
     "7977141b6ba335855bc012f79166e751c48b5477f6ef8bd50f8eb4470de620cc"),
    ("threeblock_final (the tuned 3-block CNN)", "cnn_3block_confirm_1",
     "e0c211fcced3f7a3cd9a1328d31f8677c849e04c6e9e204b6bbca644e9fb06b3"),
    ("cnn_transformer_final_1 (CNN + Transformer)", "cnn_transformer_confirm_1",
     "47ce7034379e8cc108824ad80d58623fac4bc3b9b94952d360f7cde33ea1e3e7"),
    ("lif_final (CNN + LIF, hybrid)", "cnn_lif_confirm_1",
     "553e54c97626ccce50a4e56c2c9f58e90de0ec94859090dc37561cd44ba39b9e"),
    ("snn_final (CNN + LIF, fully spiking)", "snn_confirm_1",
     "396d6ef8c2427d962c54294d237d5bd3169e6f41a1fbb873bf0e5f90a228ad02"),
)
# the screens' bases: the CNN + recurrent family (night 25b) and the heads on WavLM Large's frames (night 27)
SCREENS = ("crnn_screen_1", "heads27_pool", "heads27_transformer", "heads27_lif", "heads27_rnn")


def configurations() -> list:
    """[(name, sweep id, configuration hash, settings)] without torch: the records', then the screens' bases."""
    from affect.sweep.design import expand

    catalog.load_all()
    out = []
    for name, sweep_id, wanted in RECORDS:
        jobs, _ = expand(json.loads((paths.repo_root() / "sweeps" / f"{sweep_id}.json").read_text()))
        hits = {j["config_hash"]: j["settings"] for j in jobs if j["config_hash"] == wanted}
        if len(hits) != 1:
            raise ValueError(f"{sweep_id}: {len(hits)} configurations with hash {wanted[:12]}")
        out.append((name, sweep_id, wanted, hits[wanted]))
    for sweep_id in SCREENS:
        sweep = json.loads((paths.repo_root() / "sweeps" / f"{sweep_id}.json").read_text())
        base_stage = [s["name"] for s in sweep["stages"] if s["kind"] == "base"]
        if len(base_stage) != 1:
            raise ValueError(f"{sweep_id}: {len(base_stage)} base stages")
        jobs, _ = expand(sweep)
        hits = {j["config_hash"]: j["settings"] for j in jobs if j["stage"] == base_stage[0]}
        if len(hits) != 1:
            raise ValueError(f"{sweep_id}: {len(hits)} base configurations")
        (config, settings), = hits.items()
        out.append((f"{sweep_id} base", sweep_id, config, settings))
    return out


def shape_of(value):
    import torch

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape_of(v) for v in value]
    if value is None:
        return None
    return type(value).__name__


def trained_parameters(sweep_id: str, config: str):
    """n_parameters of the configuration's runs on this machine (efficiency.json), or None when none is here."""
    root = paths.runs_dir(sweep_id)
    found = set()
    if root.is_dir():
        for d in sorted(root.glob(f"{config[:12]}_*")):
            f = d / "efficiency.json"
            if f.is_file() and ".failed-" not in d.name and not d.name.endswith(".tmp"):
                found.add(json.loads(f.read_text()).get("n_parameters"))
    return sorted(found) if found else None


def describe(name: str, sweep_id: str, config: str, settings: dict, seconds: int = 3, sample_rate: int = 16000) -> dict:
    import torch

    from affect.data.iemocap import CLASS_ORDER
    from affect.registry import FEATURES, MODELS
    from affect.runs.spec import check_job, config_hash, input_width, validate_job

    job = validate_job(settings)
    check_job(job, sample_rate)
    assert config_hash(job) == config, (name, config_hash(job), config)
    n_classes = len(CLASS_ORDER[job.data.label_set])
    width = input_width(job)
    torch.manual_seed(0)
    model = MODELS.get(job.model_name).target(job.model, width, n_classes)
    model.eval()
    if job.features_name == "logmel":
        features = FEATURES.get("logmel").target(job.features, sample_rate, torch.device("cpu"))
        wave = 0.1 * torch.randn(1, seconds * sample_rate, generator=torch.Generator().manual_seed(0))
        x = features(wave)
        valid = torch.tensor([int(features.frames(seconds * sample_rate))])
        input_text = (f"log-mel image [1, {x.shape[1]}, {x.shape[2]}] of {seconds} s ({x.shape[2]} frames, "
                      f"{int(valid[0])} real)")
    else:
        from affect.features import frames as frame_features

        frame_rate = frame_features.frame_rate(job.features)
        frames = seconds * frame_rate
        x = torch.randn(1, width, frames, generator=torch.Generator().manual_seed(0))
        valid = torch.tensor([frames])
        input_text = (f"frames [1, {width}, {frames}] of {seconds} s (the encoder's width by {frame_rate} frames a "
                      "second)")
    calls = {}

    def hook(layer_name):
        def record(module, inputs, output):
            entry = calls.setdefault(layer_name, {"calls": 0})
            entry["calls"] += 1
            if entry["calls"] == 1:
                entry["input"] = shape_of(inputs)
                entry["output"] = shape_of(output)
        return record

    layers = []
    for layer_name, module in model.named_modules():
        if layer_name and not list(module.children()):
            module.register_forward_hook(hook(layer_name))
            layers.append((layer_name, type(module).__name__,
                           int(sum(p.numel() for p in module.parameters(recurse=False)))))
    with torch.no_grad():
        logits = model(x, valid)
    parts = {child: int(sum(p.numel() for p in module.parameters())) for child, module in model.named_children()}
    total = int(sum(p.numel() for p in model.parameters()))
    return {"name": name, "sweep": sweep_id, "configuration": config, "model": job.model_name,
            "features": job.features_name, "settings": settings, "n_classes": n_classes, "input_width": width,
            "input": input_text, "logits": shape_of(logits), "parameters": total, "parameters_by_part": parts,
            "trained_parameters": trained_parameters(sweep_id, config), "module_tree": str(model),
            "layers": [{"layer": n, "type": t, "parameters": p, **calls.get(n, {"calls": 0})} for n, t, p in layers]}


def render(d: dict) -> str:
    lines = [f"=== {d['name']}", f"sweep {d['sweep']}, configuration {d['configuration']}",
             f"model {d['model']}, features {d['features']}, {d['n_classes']} classes, input width {d['input_width']}",
             "model settings: " + json.dumps({k: v for k, v in d["settings"]["model"].items()}),
             "feature settings: " + json.dumps(d["settings"]["features"]),
             f"input: {d['input']}; logits {d['logits']}",
             f"parameters: {d['parameters']:,} (" + ", ".join(f"{k} {v:,}" for k, v in d["parameters_by_part"].items())
             + ")" + (f"; the trained runs here: {d['trained_parameters']}" if d["trained_parameters"] else
                      "; no trained run of it on this machine"),
             "", "module tree:", d["module_tree"], "", "layers (first call: input -> output; parameters; calls):"]
    for r in d["layers"]:
        lines.append(f"  {r['layer']:40s} {r['type']:22s} {json.dumps(r.get('input'))} -> {json.dumps(r.get('output'))}"
                     f"  {r['parameters']:,}  x{r['calls']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.artifacts_dir() / "architectures" / f"architectures_{stamp}"
    out.mkdir(parents=True)
    results, failures, text = [], [], []
    for name, sweep_id, config, settings in configurations():
        try:
            d = describe(name, sweep_id, config, settings)
            results.append(d)
            text.append(render(d))
            check = ("" if d["trained_parameters"] is None else
                     " (= the trained runs)" if d["trained_parameters"] == [d["parameters"]] else
                     f" (DIFFERS from the trained runs: {d['trained_parameters']})")
            print(f"{name}: {d['parameters']:,} parameters{check}, logits {d['logits']}")
        except Exception:
            failures.append({"name": name, "error": traceback.format_exc()})
            text.append(f"=== {name}\nFAILED\n{traceback.format_exc()}\n")
            print(f"{name}: FAILED\n{traceback.format_exc()}")
    (out / "architectures.txt").write_text("\n".join(text))
    (out / "architectures.json").write_text(json.dumps({"results": results, "failures": failures}, indent=1) + "\n")
    with open(out / "parameters.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["name", "sweep", "configuration", "model", "parameters", "trained_parameters"])
        for d in results:
            w.writerow([d["name"], d["sweep"], d["configuration"], d["model"], d["parameters"],
                        "" if d["trained_parameters"] is None else d["trained_parameters"]])
    print(f"{len(results)} described, {len(failures)} failed; written {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
