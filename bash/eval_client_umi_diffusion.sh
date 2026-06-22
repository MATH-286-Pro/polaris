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
default_low_level_policy_path="/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_10009.jit"
low_level_policy_paths=(
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_10009.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_9000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_8000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_7000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_6000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_4000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_2000.jit"
  "/home/ece-486/Documents/SP_PBL/polaris/robot_model_wbc/2026-05-27_02-22-04/exported/policy_1000.jit"
  )

if [ ${#low_level_policy_paths[@]} -eq 0 ]; then
  low_level_policy_paths=("${default_low_level_policy_path}")
fi

policy_train_step() {
  local policy_path="$1"
  local policy_file
  policy_file=$(basename "${policy_path}")
  if [[ "${policy_file}" =~ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

# taskname=$"UMI-FLAT-TABLE"
# taskname=$"UMI-PBL-TABLE"
# taskname=$"UMI-FLAT-TABLE-WBC"
taskname=$"UMI-PBL-TABLE-WBC"

for low_level_policy_path in "${low_level_policy_paths[@]}"; do
  train_step=$(policy_train_step "${low_level_policy_path}")

  for horizon in "${horizons[@]}"; do
    folder_name="${timestamp}_${robot_name}_h${horizon}_${FREQ_TRAJ_HIGH}hz_ll${train_step}"

    HEADLESS=0 LIVESTREAM=0 uv run scripts/eval.py \
      --environment "${taskname}" \
      --policy.client UmiGripperPos \
      --policy.port 8000 \
      --policy.open_loop_horizon "${horizon}" \
      --policy.freq_traj_high "${FREQ_TRAJ_HIGH}" \
      --policy.freq_low "${FREQ_LOW}" \
      --policy.low_level_policy_path "${low_level_policy_path}" \
      --run-folder "runs/${taskname}/${folder_name}"
      # --no-headless
  done
done
