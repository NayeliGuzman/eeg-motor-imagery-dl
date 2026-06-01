#!/bin/bash
# run_preprocessing.sh
# Usage: bash run_preprocessing.sh <input_dir> <output_dir>
#
# Run from root directory):
#   chmod +x src/preprocessing/run_preprocessing.sh
#   bash src/preprocessing/run_preprocessing.sh data/BCICIV_2a_gdf data/processed


set -e  # exit on error

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash src/preprocessing/run_preprocessing.sh <input_dir> <output_dir>"
    exit 1
fi

echo "Input directory  : $INPUT_DIR"
echo "Output directory : $OUTPUT_DIR"
echo "Starting preprocessing..."

SCRIPT_DIR=$(dirname "$0") # will find .py file where this .sh file lives
python "$SCRIPT_DIR/process_raw_data.py" "$INPUT_DIR" "$OUTPUT_DIR"

echo "Preprocessing complete."