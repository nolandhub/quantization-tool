# ABOUTME: Helpers for container-tier tests: tiny random models from shrunk real configs of the 4 verified
# ABOUTME: architectures, a tiny trained tokenizer, chat templates and @qtool event extraction.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "tests" / "fixtures" / "tiny-configs"
sys.path.insert(0, str(ROOT / "images" / "scripts"))

CORPUS = [f"calibration sentence number {i} about apples rivers and small engines" for i in range(64)]
CHAT_TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
# Mimics SmolVLM: renders only list-of-parts content, silently drops string content
PARTS_ONLY_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{% if m['content'] is not string %}"
                       "{% for p in m['content'] %}{{ p['text'] }}{% endfor %}{% endif %}</s>{% endfor %}")


def make_tokenizer(chat_template=CHAT_TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<s>", "</s>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                  pad_token="</s>")
    tok.chat_template = chat_template
    return tok


def build_model(name: str, dtype: str = "bfloat16"):
    import torch
    import transformers
    from recipe import loader_name

    d = json.loads((TINY / f"{name}.json").read_text())
    cfg = transformers.AutoConfig.for_model(**d)
    torch.manual_seed(0)
    return getattr(transformers, loader_name(d)).from_config(cfg, dtype=getattr(torch, dtype)), d




def events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        i = line.find("@qtool ")
        if i >= 0:
            out.append(json.loads(line[i + 7:]))
    return out
