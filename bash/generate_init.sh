#!/bin/bash

ENV_NAME="umi_flat_table"

python3 PolaRiS-Hub/generate_initial_conditions.py \
  --input /home/ece-486/Documents/SP_PBL/polaris/PolaRiS-Hub/${ENV_NAME}/initial_conditions_sample.json \
  --output /home/ece-486/Documents/SP_PBL/polaris/PolaRiS-Hub/${ENV_NAME}/initial_conditions.json \
  --count 40 \
  --square-center 0.3 0.0 \
  --square-size 0.25 \
  --min-distance 0.3 \
  --seed 42
