# ABOUTME: Test doubles shared by flow and CLI tests: a fake `docker run` replaying real spike logs and writing the
# ABOUTME: files real containers would (checkpoints, lm_eval results, probe event), plus common constants.
import json
import shutil

from qtool import parsers, preflight

GPU = preflight.Gpu(0, "NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16, 16376, 15928)
W4A16_QC = {"quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int", "symmetric": True,
                                                      "strategy": "group", "group_size": 128}}}}
W8A8_QC = {"quant_method": "compressed-tensors", "format": "int-quantized",
           "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"},
                                         "input_activations": {"num_bits": 8, "type": "int", "dynamic": True}}}}
FP8_DYNAMIC_QC = {"quant_method": "compressed-tensors", "format": "float-quantized",
                  "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float", "symmetric": True,
                                                            "strategy": "channel"},
                                                "input_activations": {"num_bits": 8, "type": "float", "dynamic": True,
                                                                      "strategy": "token"}}}}
NVFP4_QC = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "symmetric": True,
                                                      "strategy": "tensor_group", "group_size": 16},
                                          "input_activations": {"num_bits": 4, "type": "float", "dynamic": "local",
                                                                "strategy": "tensor_group", "group_size": 16}}}}


def ev(event, **fields):
    return "@qtool " + json.dumps({"event": event, **fields})


def arg(cmd, name):
    return cmd[cmd.index(name) + 1]


class FakeDocker:
    """Plays the role of `docker run`: replays fixture logs, writes outputs under the mounted runs dir."""

    def __init__(self, fixtures, rd):
        self.fixtures, self.rd = fixtures, rd
        self.specs = []
        self.quantize = {}  # checkpoint -> "ok" | "fail-step2" | "fail-step3" | "killed" | "no-saved"
        self.probes = {}  # model -> probe event fields (defaults: Qwen3.5-4B BF16)
        self.evals = {}  # label -> (log fixture, results dir fixture or None, exit code)

    def host(self, ctr_path):
        return self.rd / ctr_path.removeprefix("/runs/")

    def lines(self, rel):
        return parsers.split_lines((self.fixtures / rel).read_text(errors="replace"))

    def __call__(self, spec, log_path, on_line, on_tick=None):
        self.specs.append(spec)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = spec.command
        if "/qtool/probe.py" in cmd:
            model = arg(cmd, "--model")
            host = {ctr: h for h, ctr, _ in spec.mounts}.get(model, model)  # local dirs are mounted at /model
            out, code = [ev("probe", **self.probe_fields(host))], 0
        elif "/qtool/quantize.py" in cmd:
            out, code = self.run_quantize(cmd)
        else:
            out, code = self.run_eval(cmd)
        for line in out:
            on_line(line)
            if on_tick:
                on_tick()
        return code

    def probe_fields(self, model):
        base = {"model": model, "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                "dtype": "bfloat16", "multimodal": True, "remote_code": False, "quantized": False,
                "quantization_config": None, "params": 4_539_265_536, "weight_bytes": 9_078_531_072,
                "chat_mode": "chat", "chat_reason": "chat template keeps message content", "tasks_missing": []}
        return {**base, **self.probes.get(model, {})}

    def run_quantize(self, cmd):
        ckpt, mode = arg(cmd, "--checkpoint"), self.quantize.get(arg(cmd, "--checkpoint"), "ok")
        out = [ev("step", n=1),
               ev("config", architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", dtype="bfloat16",
                  multimodal=True, loader="AutoModelForImageTextToText"),
               ev("load", seconds=4.7, cls="Qwen3_5ForConditionalGeneration", params=4539265536),
               ev("recipe", ignore=["re:.*lm_head", "re:.*visual.*", "re:.*linear_attn.*"],
                  why={"re:.*lm_head": "output head", "re:.*visual.*": "vision/projector",
                       "re:.*linear_attn.*": "hybrid linear-attention / SSM mixer"},
                  sequential_target="Qwen3_5DecoderLayer", num_layers=32, num_experts=None, mtp=True, linears_total=347,
                  linears_quantized=128, quantized_sample=["model.language_model.layers.N.mlp.down_proj",
                                                           "model.language_model.layers.N.self_attn.q_proj"])]
        if ckpt != "FP8-DYNAMIC":
            out.append(ev("step", n=2))
            if mode == "fail-step2":
                return out + ["Traceback (most recent call last):", "datasets.exceptions.DatasetNotFoundError: no such dataset"], 1
            out.append(ev("calib", source="HuggingFaceH4/ultrachat_200k:train_sft", template="chat_template(str)", n=512,
                          min=199, median=1080, max=2048, truncated=48, cached=False, seconds=8.2))
        out.append(ev("step", n=3))
        if mode == "no-saved":
            return out, 0
        if mode == "fail-step3":
            return out + self.lines("llmcompressor/w4a16-error-no-cc.log"), 1
        if mode == "killed":
            return out + self.lines("llmcompressor/w4a16-fp32-oom-killed.log")[-50:], 137
        if ckpt != "FP8-DYNAMIC":
            log = "llmcompressor/w8a8-smoothquant-gptq.log" if ckpt == "INT8-W8A8" else "llmcompressor/w4a16-gptq.log"
            out += [line for line in self.lines(log) if "(" in line or "|" in line]
        tmp = self.host(arg(cmd, "--out"))
        tmp.mkdir(parents=True, exist_ok=True)
        qc = {"INT8-W8A8": W8A8_QC, "NVFP4": NVFP4_QC, "FP8-DYNAMIC": FP8_DYNAMIC_QC}.get(ckpt, W4A16_QC)
        (tmp / "config.json").write_text(json.dumps({"quantization_config": qc}))
        (tmp / "model.safetensors").write_bytes(b"x" * 1000)
        (tmp / "qtool.json").write_text(json.dumps({"signature": json.loads(arg(cmd, "--signature"))}))
        out += [ev("oneshot", seconds=905.9, peak_gpu_bytes=2097972224),
                ev("saved", path=str(tmp), bytes=5_200_000_000, cast=None, mtp=True, copied=[])]
        return out, 0

    def run_eval(self, cmd):
        label = arg(cmd, "--output_path").rsplit(".tmp-", 1)[1]
        log, results, code = self.evals[label]
        if results:
            dest = self.host(arg(cmd, "--output_path")) / "model"
            dest.mkdir(parents=True)
            src = parsers.find_results_json(self.fixtures / results)
            shutil.copy(src, dest / src.name)
        return self.lines(log), code
