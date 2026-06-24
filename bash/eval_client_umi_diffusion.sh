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

horizons=(10)
robot_name="mirror"
FREQ_TRAJ_HIGH=10
FREQ_LOW=50
timestamp=$(date +"%Y%m%d_%H%M%S")
# taskname=$"UMI-FLAT-TABLE"
# taskname=$"UMI-PBL-TABLE"
taskname=$"UMI-FLAT-TABLE-WBC"
# taskname=$"UMI-PBL-TABLE-WBC"

for horizon in "${horizons[@]}"; do
  folder_name="${timestamp}_${robot_name}_h${horizon}_${FREQ_TRAJ_HIGH}hz"

  HEADLESS=0 LIVESTREAM=0 uv run scripts/eval.py \
      --environment "${taskname}" \
      --policy.client UmiGripperPos \
      --policy.port 8000 \
      --policy.open_loop_horizon "${horizon}" \
      --policy.freq_traj_high "${FREQ_TRAJ_HIGH}" \
      --policy.freq_low "${FREQ_LOW}" \
      --run-folder "runs/${taskname}/${folder_name}" \
      # --no-headless
done
