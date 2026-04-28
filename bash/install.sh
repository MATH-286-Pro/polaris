#!/bin/bash

if ! command -v uv >/dev/null 2>&1 || ! command -v uvx >/dev/null 2>&1; then
    echo "uv or uvx is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p umi_policy
cd umi_policy
wget https://real.stanford.edu/umi/data/pretrained_models/cup_wild_vit_l_1img.ckpt

git submodule update --init --recursive
uv sync 
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

# Check NVCC (Need to have nvcc --version == cu13)
nvcc --version
