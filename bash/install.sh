#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROBOT_DESCRIPTIONS_DIR="$(dirname -- "${REPO_ROOT}")/robot_descriptions"
ROBOT_DESCRIPTIONS_LINK="${REPO_ROOT}/robot_descriptions"
ROBOT_DESCRIPTIONS_REPO="git@github.com:MATH-286-Pro/robot_descriptions.git"

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1 || ! command -v uvx >/dev/null 2>&1; then
    echo "uv or uvx is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p umi_policy
cd umi_policy
wget https://real.stanford.edu/umi/data/pretrained_models/cup_wild_vit_l_1img.ckpt

git submodule update --init --recursive

if [ -d "${ROBOT_DESCRIPTIONS_DIR}/.git" ]; then
    echo "Updating robot_descriptions in ${ROBOT_DESCRIPTIONS_DIR}..."
    git -C "${ROBOT_DESCRIPTIONS_DIR}" pull --ff-only
elif [ -e "${ROBOT_DESCRIPTIONS_DIR}" ]; then
    echo "Error: ${ROBOT_DESCRIPTIONS_DIR} exists but is not a Git repository." >&2
    exit 1
else
    echo "Cloning robot_descriptions into ${ROBOT_DESCRIPTIONS_DIR}..."
    git clone "${ROBOT_DESCRIPTIONS_REPO}" "${ROBOT_DESCRIPTIONS_DIR}"
fi

if [ -L "${ROBOT_DESCRIPTIONS_LINK}" ]; then
    ln -sfn ../robot_descriptions "${ROBOT_DESCRIPTIONS_LINK}"
elif [ -e "${ROBOT_DESCRIPTIONS_LINK}" ]; then
    echo "Error: ${ROBOT_DESCRIPTIONS_LINK} exists and is not a symbolic link." >&2
    exit 1
else
    ln -s ../robot_descriptions "${ROBOT_DESCRIPTIONS_LINK}"
fi

uv sync 
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

# Check NVCC (Need to have nvcc --version == cu13)
nvcc --version
