#!/bin/bash
git submodule update --init --recursive
uv sync 
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

# Check NVCC (Need to have nvcc --version == cu13)
nvcc --version
