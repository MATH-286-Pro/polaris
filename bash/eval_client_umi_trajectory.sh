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

HEADLESS=0 LIVESTREAM=0 uv run scripts/eval.py \
  --environment PBL-UMI \
  --policy.client UmiTrajectory \
  --policy.trajectory-path /home/ece-486/Documents/SP_PBL/umi-on-legs/data/tossing_converted.pkl \
  --policy.trajectory-index 0 \
  --policy.trajectory-control-dt 0.06666666666666667 \
  --policy.trajectory-stride 1 \
  --run-folder "runs/umi_trajectory_test/${timestamp}" \
  --no-headless
