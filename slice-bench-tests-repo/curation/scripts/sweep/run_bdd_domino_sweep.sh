#!/bin/bash

json_folders=("clean_copy/8_bdd_color" "clean_copy/8_bdd_bbox_ar" "clean_copy/8_bdd_bbox_size" "clean_copy/semantic/bdd_customs/scene/output" "clean_copy/semantic/bdd_customs/timeofday/output" "clean_copy/semantic/bdd_customs/weather/output")
counter=0
max_processes=2 # Limit to the number of GPU's you haveo

PYTHON_CMD="/data/home/jxu680/miniconda3/envs/domino/bin/python3"
PADDING=50

# Iterate over each word
for word in "${json_folders[@]}"; do
    modulo=$((counter % 2))
    cuda_device="cuda:${modulo}"

    $PYTHON_CMD domino_eval.py --json_folder "${word}" --model clip16 --padding $PADDING --device "${cuda_device}" --force &
    ((counter++))

    # Check if the maximum number of background processes has been reached
    if (( counter % max_processes == 0 )); then
        # Wait for the background processes to finish before starting the next batch
        echo "Waiting for background processes to finish..."
        wait
        echo "Background processes finished.. Moving on to the next batch."
    fi
done

