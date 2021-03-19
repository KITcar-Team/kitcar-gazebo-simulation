#!/bin/bash
OUTPUT_DIR=$3
ROAD=$4

for i in $1; do
  for j in $2; do
    python3 -m simulation.utils.machine_learning.data.extract_simulated_images \
    --road "$ROAD" \
    --seed "KITCAR$i$j" \
    --output_dir "$OUTPUT_DIR" \
    --image_topic /camera/image_raw \
    --label_camera \
    --control_sim_rate \
    --label_image_topic /simulation/label_camera/image \
    --label_topic /simulation/label_camera/image_labels \
    --apply_gan \
    --label_file "$OUTPUT_DIR"/labels.yaml \
    --factor_keep_pixels "$i" \
    --factor_keep_colored_pixels "$j";
  done
done

# Remove duplicates
fdupes -dN "$OUTPUT_DIR"/
fdupes -dN "$OUTPUT_DIR"/debug
python3 -c "from simulation.utils.machine_learning.data import LabeledDataset;LabeledDataset.filter_file(\"$OUTPUT_DIR/labels.yaml\")"
