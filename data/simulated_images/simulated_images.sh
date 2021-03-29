#!/bin/bash

SEED=$1
N=$2
OUTPUT_DIR=$3
EXTRA_ARGS=$4

for i in $(seq 1 "$N") ; do
    python3 -m simulation.utils.machine_learning.data.extract_simulated_images --seed "$SEED$i" --output_dir "$OUTPUT_DIR" --image_topic /camera/image_raw "$EXTRA_ARGS";
done

# Remove duplicates
fdupes -dNq "$OUTPUT_DIR"/
