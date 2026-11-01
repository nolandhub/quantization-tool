# ABOUTME: Generic recipe derivation for any transformers model (spike Q9): loader class, ignore list,
# ABOUTME: sequential layer, MTP, remote-code and chat-template checks. Config helpers take plain dicts; torch is lazy.
import re

VISION_PREFIXES = ("visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector",
                   "mm_projector", "connector", "image_newline")
MIXERS = ("linear_attn", "mamba", "mixer", "ssm")
ATTENTION_TYPES = ("full_attention", "attention", "sliding_attention")
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def orig_dtype(cfg: dict) -> str | None:
    for c in (cfg, text_cfg(cfg)):
        d = c.get("dtype") or c.get("torch_dtype")
        if d:
            return str(d).replace("torch.", "")
    return None


def is_multimodal(cfg: dict) -> bool:
    return "vision_config" in cfg


def loader_name(cfg: dict) -> str:
    return "AutoModelForImageTextToText" if is_multimodal(cfg) else "AutoModelForCausalLM"


def num_experts(cfg: dict) -> int | None:
    for k in EXPERT_KEYS:
        v = text_cfg(cfg).get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 1:
            return v
    return None


def is_hybrid(cfg: dict) -> bool:
    return any(t not in ATTENTION_TYPES for t in text_cfg(cfg).get("layer_types") or [])


def mtp_present(cfg: dict) -> bool:
    return any(("mtp" in k or "nextn" in k) and isinstance(v, int) and not isinstance(v, bool) and v > 0
               for k, v in text_cfg(cfg).items())


def remote_code_required(cfg: dict, transformers_module) -> bool:
    archs = cfg.get("architectures") or []
    return "auto_map" in cfg and not any(hasattr(transformers_module, a) for a in archs)


def vision_segment(name: str) -> str | None:
    for seg in name.split("."):
        if seg.startswith(VISION_PREFIXES):
            return seg
    return None


def _ignored(name: str, patterns: list[str]) -> bool:
    return any(re.match(p[3:], name) if p.startswith("re:") else name == p for p in patterns)


def derive(model, cfg: dict, extra_ignore=()) -> dict:
    import torch

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    ignore, why = ["re:.*lm_head"], {"re:.*lm_head": "output head"}

    for seg in sorted({s for n, _ in model.named_modules() if (s := vision_segment(n))}):
        rx = f"re:.*{re.escape(seg)}.*"
        ignore.append(rx)
        why[rx] = "vision/projector"

    ne = num_experts(cfg)
    if ne:
        for suffix in sorted({re.sub(r"^.*?layers\.\d+\.", "", n) for n, m in linears
                              if m.out_features in (ne, 1) and not vision_segment(n)}):
            rx = f"re:.*{re.escape(suffix)}$"
            ignore.append(rx)
            why[rx] = f"MoE router (out_features in {{{ne}, 1}})"

    if is_hybrid(cfg):
        for p in MIXERS:
            if any(f".{p}." in n for n, _ in linears):
                rx = f"re:.*{p}.*"
                ignore.append(rx)
                why[rx] = "hybrid linear-attention / SSM mixer"

    for p in extra_ignore:
        if p not in ignore:
            ignore.append(p)
            why[p] = "--ignore"

    # Largest non-vision ModuleList whose element class is in _no_split_modules; size alone picks MoE expert lists
    no_split = set(getattr(model, "_no_split_modules", None) or [])
    lists = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) > 0 and not vision_segment(n)]
    preferred = [x for x in lists if type(x[1][0]).__name__ in no_split] or lists
    best = max(preferred, key=lambda x: len(x[1]), default=None)

    quantized = [n for n, _ in linears if not _ignored(n, ignore)]
    return {
        "ignore": ignore,
        "why": why,
        "sequential_target": type(best[1][0]).__name__ if best else None,
        "num_layers": len(best[1]) if best else None,
        "num_experts": ne,
        "mtp": mtp_present(cfg),
        "linears_total": len(linears),
        "linears_quantized": len(quantized),
        "quantized_sample": sorted({re.sub(r"\.\d+\.", ".N.", n) for n in quantized})[:12],
    }


def keeps_content(rendered: str, messages: list[dict]) -> bool:
    # Some templates (SmolVLM) silently drop string content; every message text must survive rendering
    return all(m["content"][:40] in rendered for m in messages if isinstance(m.get("content"), str) and m["content"])


def _as_parts(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


def render_calib(tok, conversations: list[list[dict]]) -> tuple[str, list[str]]:
    """Return (mode, texts): 'chat_template(str)', 'chat_template(parts)' or 'raw' (template missing or drops text)."""
    if getattr(tok, "chat_template", None):
        as_str = [tok.apply_chat_template(c, tokenize=False) for c in conversations]
        if all(keeps_content(r, c) for r, c in zip(as_str[:8], conversations[:8])):
            return "chat_template(str)", as_str
        as_parts = [tok.apply_chat_template(_as_parts(c), tokenize=False) for c in conversations]
        if all(keeps_content(r, c) for r, c in zip(as_parts[:8], conversations[:8])):
            return "chat_template(parts)", as_parts
    return "raw", ["\n\n".join(m["content"] for m in c) for c in conversations]


def eval_chat_mode(tok) -> tuple[str, str]:
    """lm_eval sends string content; chat mode only if the template keeps it. Returns (mode, reason)."""
    sample = [{"role": "user", "content": "Probe question: what is two plus three?"},
              {"role": "assistant", "content": "Probe answer: two plus three is five."}]
    if not getattr(tok, "chat_template", None):
        return "completion", "tokenizer has no chat template"
    if not keeps_content(tok.apply_chat_template(sample, tokenize=False), sample):
        return "completion", "chat template drops string message content"
    return "chat", "chat template keeps message content"
