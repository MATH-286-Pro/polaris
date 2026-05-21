#!/bin/bash

# In a separate process, start evaluation process
# sudo apt install ffmpeg # for saving videos

uv run scripts/eval.py \
    --environment PBL-TEST \
    --policy.port 8000 \
    --run-folder runs/PBL-TEST

# uv run scripts/eval.py \
#     --environment DROID-FoodBussing \
#     --policy.port 8000 \
#     --run-folder runs/DROID-FoodBussing
