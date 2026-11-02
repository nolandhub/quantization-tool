# ABOUTME: Static tables from spec §3: the 6 schemes, the 3 checkpoints they share, and the eval capability
# ABOUTME: matrix per GPU compute capability (SM). Pure data plus lookups; every hardware rule lives here.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scheme:
    name: str
    checkpoint: str | None  # None: no checkpoint, vLLM quantizes the BF16 model online
    row: str  # capability-matrix row
    env: dict = field(default_factory=dict)
    quantization: str | None = None


SCHEMES = {s.name: s for s in [
    Scheme("int4-w4a16", "INT4-W4A16", "int4-w4a16"),
    Scheme("int4-w4a8-fp8", "INT4-W4A16", "int4-w4a8", {"VLLM_MARLIN_INPUT_DTYPE": "fp8"}),
    Scheme("int4-w4a8-int8", "INT4-W4A16", "int4-w4a8", {"VLLM_MARLIN_INPUT_DTYPE": "int8"}),
    Scheme("int8-w8a8", "INT8-W8A8", "int8-w8a8"),
    Scheme("nvfp4-w4a4", "NVFP4", "nvfp4-w4a4"),
    Scheme("fp8-w8a8", None, "fp8-w8a8", quantization="fp8"),
]}

# llmcompressor preset per checkpoint and how many sequential passes oneshot makes (SmoothQuant adds one)
CHECKPOINTS = {
    "INT4-W4A16": {"preset": "W4A16", "recipe": "GPTQModifier targets=Linear scheme=W4A16 dampening_frac=0.05", "passes": 1},
    "INT8-W8A8": {"preset": "W8A8", "passes": 2,
                  "recipe": "SmoothQuantModifier smoothing_strength=0.8 + GPTQModifier targets=Linear scheme=W8A8"},
    "NVFP4": {"preset": "NVFP4", "recipe": "QuantizationModifier targets=Linear scheme=NVFP4", "passes": 1},
}

OK, WARN, NO = "ok", "warn", "no"
NO_SM70 = (NO, "vLLM 0.28 has no SM 7.0 kernels")
W4A8_SM = (NO, "Marlin W4A8 needs SM 8.9 or 12.0")
NO_INT8 = (NO, "no INT8 kernels on SM >= 10.0 (vLLM docs)")
FP4_WEIGHT_ONLY = (WARN, "runs as weight-only FP4 (Marlin), activations stay BF16")
FP32_SLOW = (WARN, "float32 compute, about 10x slower than bfloat16")
MATRIX = {
    "int4-w4a16": {"7.0": NO_SM70, "7.5": (NO, "INT4 needs SM > 8.0 (vLLM docs)"), "8.x": (OK, ""), "8.9": (OK, ""),
                   "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "int4-w4a8": {"7.0": NO_SM70, "7.5": W4A8_SM, "8.x": W4A8_SM, "8.9": (OK, ""),
                  "9.0": (NO, "Marlin W4A8 is slower than W4A16 on SM 9.0"), "10.x": W4A8_SM, "12.x": (OK, "")},
    "int8-w8a8": {"7.0": NO_SM70, "7.5": (OK, ""), "8.x": (OK, ""), "8.9": (OK, ""), "9.0": (OK, ""),
                  "10.x": NO_INT8, "12.x": NO_INT8},
    "nvfp4-w4a4": {"7.0": NO_SM70, "7.5": FP4_WEIGHT_ONLY, "8.x": FP4_WEIGHT_ONLY, "8.9": FP4_WEIGHT_ONLY,
                   "9.0": FP4_WEIGHT_ONLY, "10.x": (OK, ""), "12.x": (OK, "")},
    "fp8-w8a8": {"7.0": NO_SM70, "7.5": (NO, "FP8 needs SM >= 8.0"), "8.x": (WARN, "runs as W8A16 weight-only (Marlin)"),
                 "8.9": (OK, ""), "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "baseline": {"7.0": NO_SM70, "7.5": (WARN, "vLLM forces FP16 (no BF16 on SM 7.5)"), "8.x": (OK, ""), "8.9": (OK, ""),
                 "9.0": (OK, ""), "10.x": (OK, ""), "12.x": (OK, "")},
    "quantize": {"7.0": FP32_SLOW, "7.5": FP32_SLOW, "8.x": (OK, ""), "8.9": (OK, ""), "9.0": (OK, ""),
                 "10.x": (OK, ""), "12.x": (OK, "")},
}
VERIFIED = {(row, "8.9") for row in MATRIX} | {("quantize", "7.0")}
# Preferred replacement when a scheme cannot run on a GPU (first one that is OK there wins)
ALTERNATIVES = {"int8-w8a8": ["fp8-w8a8", "int4-w4a16"], "fp8-w8a8": ["int8-w8a8", "int4-w4a16"],
                "int4-w4a8": ["int4-w4a16", "int8-w8a8"], "nvfp4-w4a4": ["int4-w4a16"],
                "int4-w4a16": ["int8-w8a8", "fp8-w8a8"]}
GROUPS = {"build a checkpoint": ["int4-w4a16", "int8-w8a8", "nvfp4-w4a4"],
          "reuse INT4-W4A16": ["int4-w4a8-fp8", "int4-w4a8-int8"], "no checkpoint": ["fp8-w8a8"]}


@dataclass(frozen=True)
class Support:
    status: str
    note: str
    verified: bool


def _cc(cc: str) -> tuple[int, int]:
    major, minor = cc.split(".")
    return int(major), int(minor)


def sm_column(cc: str) -> str:
    major, minor = _cc(cc)
    if major < 7 or (major == 7 and minor < 5):
        return "7.0"
    if major == 7:
        return "7.5"
    if major == 8:
        return "8.9" if minor >= 9 else "8.x"
    if major == 9:
        return "9.0"
    if major in (10, 11):
        return "10.x"
    return "12.x"


def support(row: str, cc: str) -> Support:
    col = sm_column(cc)
    status, note = MATRIX[row][col]
    return Support(status, note, (row, col) in VERIFIED)


def alternative(scheme: str, cc: str) -> str | None:
    for name in ALTERNATIVES.get(SCHEMES[scheme].row, []):
        if support(SCHEMES[name].row, cc).status == OK:
            return name
    return None


def compute_dtype(cc: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "bfloat16" if _cc(cc)[0] >= 8 else "float32"  # never float16: BF16-trained models can overflow it


def scheme_help() -> str:
    return (
        "schemes that build a checkpoint:\n"
        "  int4-w4a16      INT4 weights, 16-bit activations (checkpoint INT4-W4A16)\n"
        "  int8-w8a8       INT8 weights and activations (checkpoint INT8-W8A8)\n"
        "  nvfp4-w4a4      FP4 weights and activations (checkpoint NVFP4)\n"
        "schemes that reuse INT4-W4A16 (quantized once, no extra checkpoint):\n"
        "  int4-w4a8-fp8   INT4 weights, FP8 activations at runtime (Marlin)\n"
        "  int4-w4a8-int8  INT4 weights, INT8 activations at runtime (Marlin)\n"
        "no checkpoint, vLLM quantizes at load time:\n"
        "  fp8-w8a8        FP8 weights and activations"
    )


def parse_schemes(text: str) -> list[str]:
    names = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in SCHEMES:
            raise ValueError(f"unknown scheme '{name}'.\n" + scheme_help())
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("--scheme is empty")
    return names


def checkpoints_for(schemes: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in schemes:
        ckpt = SCHEMES[s].checkpoint
        if ckpt:
            out.setdefault(ckpt, []).append(s)
    return out
