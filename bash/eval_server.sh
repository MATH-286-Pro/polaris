#!/bin/bash

# # Starting from the root of this repo. This will setup openpi and host a pi05 policy.
# cd third_party/openpi
# # GIT_LFS_SKIP_SMUDGE=1 uv sync
# # GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run scripts/serve_policy.py \
#     --port 8000 policy:checkpoint \
#     --policy.config pi05_droid_jointpos_polaris \
#     --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris

uv run src/polaris/policy/umi_diffusion_server.py \
  --port 8000 \
  --checkpoint /home/ece-486/Documents/SP_PBL/umi/data/outputs/umi/cup_wild_vit_l_1img.ckpt \
  --device cuda  
