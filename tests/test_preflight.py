# ABOUTME: Tests for qtool.preflight on real nvidia-smi fixtures (rtx4070, 8x V100 Apollo): GPU selection,
# ABOUTME: process labels, VRAM/RAM thresholds, eval capability by SM, eval memory budget, --act and the run lock.
import os

import pytest

from qtool import preflight as pf

GB = 10 ** 9


def gpus_of(fixtures, name):
    return pf.parse_gpus((fixtures / f"nvidia-smi/{name}.csv").read_text())


def apps_of(fixtures, name, uuid):
    return pf.parse_apps((fixtures / f"nvidia-smi/{name}.csv").read_text(),
                         (fixtures / f"nvidia-smi/{uuid}.csv").read_text())


def test_parse_rtx4070(fixtures):
    assert gpus_of(fixtures, "rtx4070") == [pf.Gpu(0, "NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16, 16376, 15928)]
    assert apps_of(fixtures, "rtx4070-apps", "rtx4070-uuid") == []
    assert apps_of(fixtures, "rtx4070-busy-apps", "rtx4070-uuid") == [pf.GpuApp(0, 3551455, "python", 1524)]


def test_parse_apollo_maps_uuid_to_index(fixtures):
    apps = apps_of(fixtures, "apollo-8gpu-apps", "apollo-8gpu-uuid")
    assert len(gpus_of(fixtures, "apollo-8gpu")) == 8 and len(apps) == 9
    assert pf.GpuApp(4, 2273897, "VLLM::EngineCore", 30706) in apps


def test_parse_gpus_non_numeric_row_is_preflight_error():
    row = "0, NVIDIA X, 8.9, [N/A], [N/A], [N/A]"
    with pytest.raises(pf.PreflightError) as e:
        pf.parse_gpus(row + "\n")
    assert str(e.value) == f"cannot parse nvidia-smi GPU row: {row!r}"


def test_parse_gpus_short_row_is_preflight_error():
    with pytest.raises(pf.PreflightError, match=r"cannot parse nvidia-smi GPU row: '0, NVIDIA X, 8.9'"):
        pf.parse_gpus("0, NVIDIA X, 8.9\n")


def test_parse_apps_tolerates_unsupported_memory_and_skips_bad_rows():
    uuid = "0, GPU-aaa\n"
    apps = ("GPU-aaa, 123, python, [N/A]\n"  # memory unsupported: kept with 0 MiB
            "GPU-aaa, [N/A], python, 5\n"    # pid unsupported: skipped
            "GPU-aaa, 124\n"                 # too few columns: skipped
            "GPU-aaa, 125, python, 64\n")
    assert pf.parse_apps(apps, uuid) == [pf.GpuApp(0, 123, "python", 0), pf.GpuApp(0, 125, "python", 64)]


def test_single_gpu_is_chosen_without_flag(fixtures):
    assert pf.select_gpu(gpus_of(fixtures, "rtx4070"), None).index == 0


def test_multi_gpu_requires_flag_and_prints_table(fixtures):
    gpus, apps = gpus_of(fixtures, "apollo-8gpu"), apps_of(fixtures, "apollo-8gpu-apps", "apollo-8gpu-uuid")
    with pytest.raises(pf.PreflightError) as e:
        pf.select_gpu(gpus, None, apps)
    msg = str(e.value)
    assert msg.startswith("this machine has 8 GPUs: choose one with --gpu N")
    assert "pid 2273897 VLLM::EngineCore 30706 MiB" in msg
    assert pf.select_gpu(gpus, 3).index == 3
    with pytest.raises(pf.PreflightError, match="--gpu 9 does not exist"):
        pf.select_gpu(gpus, 9)


def test_query_gpus_reports_nvidia_smi_failure():
    class R:
        returncode, stdout, stderr = 9, "", "NVIDIA-SMI has failed"
    with pytest.raises(pf.PreflightError, match="nvidia-smi failed: NVIDIA-SMI has failed"):
        pf.query_gpus(run=lambda *a, **k: R())


def test_container_label_from_cgroup():
    cid = "42feb688a9f21aab9a07719310988c7d71d4e1bded2f9e5d709712f6d86274ac"
    text = f"0::/system.slice/docker-{cid}.scope\n"
    assert pf.container_id_from_cgroup(text) == cid
    assert pf.container_id_from_cgroup("12:pids:/docker/" + cid) == cid
    app = pf.GpuApp(0, 3551455, "python", 1524)
    assert pf.process_label(app, {cid: "trt-export-server"}, read=lambda pid: text) == \
        "pid 3551455 (container trt-export-server)"
    assert pf.process_label(app, {}, read=lambda pid: "0::/user.slice") == "pid 3551455 python"


def test_mem_available():
    assert pf.mem_available_bytes("MemTotal: 30000000 kB\nMemAvailable:   27100000 kB\n") == 27100000 * 1024
    with pytest.raises(pf.PreflightError):
        pf.mem_available_bytes("MemTotal: 1 kB\n")


def test_quantize_thresholds_bf16_pass_on_rtx4070(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    params_4b = 4_539_265_536
    assert pf.check_quantize(gpu, params_4b, "bfloat16", 27 * GB, ["int4-w4a16"], no_eval=False) == []


def test_quantize_fp32_4b_blocked_by_ram(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 27 * GB, ["int4-w4a16"], no_eval=False)
    assert [f.level for f in found] == ["error"]
    assert found[0].message == "host has 27.0 GB RAM available, quantize needs 31.8 GB (3.5 x 9.1 GB float32)"


def test_quantize_vram_error_when_gpu_busy(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[4]  # 1.7 GiB free
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 700 * GB, ["int4-w4a16"], no_eval=True)
    assert [f.level for f in found] == ["warn", "error"]
    assert found[0].message == "quantize on SM 7.0: float32 compute, about 10x slower than bfloat16"
    assert found[1].message.startswith("GPU 4 has 1.9 GB free VRAM, quantize needs 9.1 GB")


def test_skip_preflight_drops_thresholds_only(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[4]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 1 * GB, ["int4-w4a16"], no_eval=False, thresholds=False)
    assert [f.level for f in found] == ["warn", "error"] and "add --no-eval" in found[1].message


def test_v100_without_no_eval_is_an_error(fixtures):
    gpu = gpus_of(fixtures, "apollo-8gpu")[1]
    found = pf.check_quantize(gpu, 4_539_265_536, "float32", 700 * GB, ["int4-w4a16"], no_eval=False)
    assert found[-1].level == "error" and "add --no-eval" in found[-1].message


def test_eval_sm_errors_and_warnings():
    sm10 = pf.Gpu(0, "B200", "10.0", 0, 180000, 180000)
    msgs = [(f.level, f.message) for f in pf.check_eval_sm(sm10, ["int8-w8a8"])]
    assert ("error", "int8-w8a8 is not supported on SM 10.0: no INT8 kernels on SM >= 10.0 (vLLM docs). Use fp8-w8a8.") in msgs
    sm86 = pf.Gpu(0, "RTX 3090", "8.6", 0, 24576, 24576)
    assert ("warn", "fp8-w8a8 on SM 8.6 runs as W8A16 weight-only (Marlin) [per vLLM docs, unverified]") in \
        [(f.level, f.message) for f in pf.check_eval_sm(sm86, ["fp8-w8a8"])]


def test_eval_sm_89_is_verified_and_quiet(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    assert pf.check_eval_sm(gpu, ["int4-w4a16", "int4-w4a8-fp8", "int4-w4a8-int8", "int8-w8a8", "fp8-w8a8"]) == []


def test_eval_fit_olmoe_bf16_skipped_quantized_fits(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    reason = pf.check_eval_fit("bf16 baseline", int(12.89 * pf.GIB), gpu)
    assert reason == "bf16 baseline does not fit: 13.8 GB weights > 13.5 GB budget (0.85 x 16.0 GiB - 1 GiB)"
    assert pf.check_eval_fit("int4-w4a16", int(3.7 * GB), gpu) is None
    assert pf.check_eval_fit("x", None, gpu) is None


def test_eval_free_vram(fixtures):
    assert pf.check_eval_free(gpus_of(fixtures, "rtx4070")[0]) == []
    busy = pf.Gpu(0, "x", "8.9", 8000, 16376, 8376)
    assert "vLLM needs 13.6 GiB free on GPU 0" in pf.check_eval_free(busy)[0].message


W4A16_QC = {"quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                                                      "group_size": 128}}}}


def test_act_checks(fixtures):
    gpu = gpus_of(fixtures, "rtx4070")[0]
    assert pf.check_act(gpu, W4A16_QC) == []
    w8 = {"quant_method": "compressed-tensors",
          "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"}}}}
    assert pf.check_act(gpu, w8)[0].message == \
        "--act needs an INT4 symmetric group-wise checkpoint, this model is compressed-tensors int8 symmetric=True strategy=channel"
    a100 = pf.Gpu(0, "A100", "8.0", 0, 40000, 40000)
    assert pf.check_act(a100, W4A16_QC)[0].message.startswith("--act needs SM 8.9 or 12.0")
    assert pf.is_int4_sym_group(None) == (False, "not quantized")


def test_run_lock(tmp_path):
    path = tmp_path / "q" / ".lock"
    lock = pf.RunLock(path)
    lock.acquire()
    assert path.read_text() == str(os.getpid())
    lock.release()
    assert not path.exists()
    path.write_text("1")  # PID 1 is always alive
    with pytest.raises(pf.PreflightError, match=r"another qtool run \(pid 1\) is active"):
        pf.RunLock(path).acquire()
    path.write_text("999999999")  # dead PID: stale lock is taken over
    pf.RunLock(path).acquire()


@pytest.mark.parametrize("content", ["-1", "0"])
def test_run_lock_ignores_non_positive_pid(tmp_path, content):
    path = tmp_path / ".lock"
    path.write_text(content)  # os.kill(-1, 0) would signal every process: never a holder
    pf.RunLock(path).acquire()
    assert path.read_text() == str(os.getpid())
