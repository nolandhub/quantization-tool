# ABOUTME: Tests for qtool.parsers against real spike fixtures: llmcompressor progress, vLLM/lm_eval eval logs,
# ABOUTME: known error lines and lm_eval results JSON. Expected numbers were read from the fixtures themselves.
import json

import pytest

from qtool import parsers


def lines_of(fixtures, rel):
    return parsers.split_lines((fixtures / rel).read_text(errors="replace"))


def feed_all(progress, lines):
    for line in lines:
        progress.feed(line)
    return progress


def test_parse_event_finds_mark_anywhere_in_line():
    assert parsers.parse_event('Dispatching: 0%|  |@qtool {"event": "saved", "bytes": 5}') == {"event": "saved", "bytes": 5}
    assert parsers.parse_event("@qtool {not json") is None
    assert parsers.parse_event('@qtool {"no": "event"}') is None
    assert parsers.parse_event("plain line") is None


def test_split_lines_handles_carriage_returns():
    assert parsers.split_lines("a\rb\r\nc\n\nd") == ["a", "b", "c", "d"]


@pytest.mark.parametrize("rel,total,passes,gptq,gsec,smoothed,linearized,module", [
    ("llmcompressor/w4a16-gptq.log", 33, 1, 128, 218.72, 0, False, "model.language_model.layers.31.mlp.down_proj"),
    ("llmcompressor/w8a8-smoothquant-gptq.log", 33, 2, 128, 204.91, 38, False, "model.language_model.layers.31.mlp.down_proj"),
    ("llmcompressor/olmoe-w4a16-generic.log", 17, 1, 3136, 2070.83, 0, True, "model.layers.15.mlp.experts.63.down_proj"),
    ("llmcompressor/smolvlm2-w4a16-generic.log", 25, 1, 168, 195.07, 0, False, "model.text_model.layers.23.mlp.down_proj"),
    ("llmcompressor/w4a16-gptq-v100-fp32.log", 33, 1, 128, 1011.44, 0, False, "model.language_model.layers.31.mlp.down_proj"),
])
def test_quantize_progress_on_real_logs(fixtures, rel, total, passes, gptq, gsec, smoothed, linearized, module):
    p = feed_all(parsers.QuantizeProgress(), lines_of(fixtures, rel))
    assert (p.subgraph, p.total, p.pass_no) == (total, total, passes)
    assert (p.done, p.samples) == (512, 512)
    assert p.gptq_modules == gptq and p.gptq_seconds == pytest.approx(gsec, abs=0.01)
    assert (p.smoothed, p.linearized, p.module) == (smoothed, linearized, module)
    assert p.fraction(passes) == 1.0


def test_fraction_is_monotonic_and_counts_both_passes(fixtures):
    p, seen, at_pass2 = parsers.QuantizeProgress(), [], None
    for line in lines_of(fixtures, "llmcompressor/w8a8-smoothquant-gptq.log"):
        if p.feed(line) and p.total:
            seen.append(p.fraction(2))
            if p.pass_no == 2 and at_pass2 is None:
                at_pass2 = seen[-1]
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert at_pass2 == pytest.approx(0.5, abs=0.01)


def test_status_text():
    p = parsers.QuantizeProgress()
    assert p.status_text() == "starting"
    p.feed("(17/33): Calibrating:  50%|█████     | 256/512 [00:06<00:06, 40.00it/s]")
    assert p.status_text() == "(17/33) Calibrating 256/512"


def test_progress_of_killed_run_stops_early(fixtures):
    p = feed_all(parsers.QuantizeProgress(), lines_of(fixtures, "llmcompressor/w4a16-fp32-oom-killed.log"))
    assert (p.subgraph, p.total, p.gptq_modules) == (2, 33, 0)


@pytest.mark.parametrize("rel,kernels,gib,kv,downcast", [
    ("lm_eval/eval-w4a16-fp8.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 5.11, 104448, None),
    ("lm_eval/eval-w8a8.log", ["CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"], 6.29, 82212, None),
    ("lm_eval/eval-fp8-online.log", ["CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod"], 5.09, 124928, None),
    ("lm_eval/eval-bf16-chat.log", [], 8.61, 42422, None),
    ("lm_eval/eval-w4a16-v100.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 5.11, 103862, "float32->bfloat16"),
    ("lm_eval/eval-phi4mini-w4a16.log", ["MarlinLinearKernel for CompressedTensorsWNA16"], 2.73, 73855, None),
])
def test_eval_progress_on_real_logs(fixtures, rel, kernels, gib, kv, downcast):
    p = feed_all(parsers.EvalProgress(), lines_of(fixtures, rel))
    assert (p.kernels, p.load_gib, p.kv_tokens, p.downcast) == (kernels, gib, kv, downcast)
    assert (p.prompts_done, p.prompts_total) == (50, 50)
    assert p.table[0].startswith("|Tasks|") and len(p.table) == 4
    assert p.table[3].startswith("|     |       |strict-match")


@pytest.mark.parametrize("rel,quote,hint", [
    ("lm_eval/eval-bf16-docs.err-mamba.log", "max_num_seqs (256) exceeds available Mamba cache blocks (124)", "Mamba"),
    ("lm_eval/eval-olmoe-bf16.err-no-kv-memory.log", "No available memory for the cache blocks", "KV cache"),
    ("lm_eval/eval-smolvlm2-bf16.err-num2words.log", "ImportError: Package `num2words` is required", "missing"),
    ("llmcompressor/w4a16-error-no-cc.log", "RuntimeError: Failed to find C compiler", ""),
    ("llmcompressor/w4a16-error-pydantic.log", "ValidationError: 1 validation error for GPTQModifier", ""),
])
def test_find_error(fixtures, rel, quote, hint):
    got = parsers.find_error(lines_of(fixtures, rel))
    assert got is not None and quote in got[0] and hint in got[1]


def test_find_error_none_when_log_just_stops(fixtures):
    assert parsers.find_error(lines_of(fixtures, "llmcompressor/w4a16-fp32-oom-killed.log")) is None


def test_read_results(fixtures):
    path = parsers.find_results_json(fixtures / "lm_eval/out/w4a16-fp8")
    r = parsers.read_results(path)
    first = parsers.primary(r["metrics"])["gsm8k"]
    assert (first.name, first.filter, first.value) == ("exact_match", "strict-match", 0.78)
    assert first.stderr == pytest.approx(0.0592, abs=1e-4) and first.label == "gsm8k strict-match"
    assert [m.filter for m in r["metrics"]] == ["strict-match", "flexible-extract"]
    assert r["n_samples"] == {"gsm8k": 50} and r["n_shot"] == {"gsm8k": 5}


def test_metrics_skip_non_numeric_stderr():
    ms = parsers.metrics_from({"t": {"acc,none": 0.5, "acc_stderr,none": "N/A", "alias": "t"}})
    assert ms == [parsers.Metric("t", "acc", "none", 0.5, None)] and ms[0].label == "t"


def test_find_results_json_missing(tmp_path):
    assert parsers.find_results_json(tmp_path) is None
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "results_2026-01-01.json").write_text(json.dumps({"results": {}}))
    assert parsers.find_results_json(tmp_path).name == "results_2026-01-01.json"
