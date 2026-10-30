# ABOUTME: Image cho step 1-3 cua qtool: transformers + llmcompressor tren torch 2.13.
# ABOUTME: CUDA=cu129 cho SM 7.5-12.0, CUDA=cu126 cho SM 5.0-9.0 (co V100); ghim theo spike 2026-10-30.
FROM python:3.11-slim
ARG CUDA=cu129
# Triton can C compiler de build helper luc chay (kernel Hessian GPTQ)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/${CUDA} && \
    pip install --no-cache-dir llmcompressor==0.13.0 transformers==5.14.1 datasets
