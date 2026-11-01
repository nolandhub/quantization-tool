# ABOUTME: Image cho step 4 cua qtool: vLLM 0.28 (giong bake-off) + lm-eval.
# ABOUTME: Ghim torch/vllm/transformers/compressed-tensors cua image goc; huggingface-hub<2 cho transformers 5.15 (spike 2026-10-30).
FROM vllm/vllm-openai:v0.28.0-cu129-ubuntu2404
RUN pip freeze | grep -iE "^(torch|vllm|transformers|compressed-tensors)==" > /tmp/vllm-constraints.txt && \
    echo "huggingface-hub<2.0" >> /tmp/vllm-constraints.txt && \
    pip install --no-cache-dir -c /tmp/vllm-constraints.txt "lm-eval[api]==0.4.13"
# Processor SmolVLM / Idefics can num2words ca luc vLLM load model (spike Q9)
RUN pip install --no-cache-dir -c /tmp/vllm-constraints.txt num2words
ENTRYPOINT []
