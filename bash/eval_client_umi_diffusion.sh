#!/bin/bash

# In a separate process, start evaluation process
# sudo apt install ffmpeg # for saving videos

# uv run scripts/eval.py \
#     --environment PBL-UMI \
#     --policy.client UmiGripperPos \
#     --policy.port 8000 \
#     --run-folder runs/PBL-UMI

# uv run scripts/eval.py \
#     --environment DROID-FoodBussing \
#     --policy.client DroidJointPos \
#     --policy.port 8000 \
#     --run-folder runs/DROID-FoodBussing

timestamp=$(date +"%Y%m%d_%H%M%S")
# taskname=$"UMI-FLAT-TABLE"
taskname=$"UMI-PBL-TABLE"

HEADLESS=0 LIVESTREAM=0 uv run scripts/eval.py \
  --environment ${taskname} \
  --policy.client UmiGripperPos \
  --policy.port 8000 \
  --policy.open_loop_horizon 8 \
  --run-folder "runs/${taskname}/${timestamp}" \
  # --no-headless
