# ABOUTME: Fixtures for container-tier tests (run inside qtool-compress, CPU only): saved tiny model dirs
# ABOUTME: and a calibration .jsonl built from the helpers in tinyhelpers.py.
import json
from pathlib import Path

import pytest

from tinyhelpers import CHAT_TEMPLATE, CORPUS, build_model, make_tokenizer


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    """Factory: tiny_dir(name, chat_template=...) -> saved model dir (weights stored as bfloat16)."""
    made = {}

    def make(name: str, chat_template=CHAT_TEMPLATE) -> Path:
        key = (name, chat_template)
        if key not in made:
            out = tmp_path_factory.mktemp(f"src-{name}")
            model, _ = build_model(name)
            model.save_pretrained(out)
            make_tokenizer(chat_template).save_pretrained(out)
            made[key] = out
        return made[key]

    return make


@pytest.fixture(scope="session")
def calib_jsonl(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("calib") / "calib.jsonl"
    with open(path, "w") as f:
        for i, text in enumerate(CORPUS):
            f.write(json.dumps({"messages": [{"role": "user", "content": text},
                                             {"role": "assistant", "content": f"answer {i} about engines"}]}) + "\n")
    return path
