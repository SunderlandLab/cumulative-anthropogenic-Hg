#!/bin/bash

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Define an array of file ID and relative output path pairs
file_pairs=(
    "10884783 ../inputs/emissions/AnthroPre1510_68Gg.tab"
    "10884782 ../inputs/emissions/SSP1-26_region_1510_2300.tab"
    "10884780 ../inputs/emissions/SSP1-26_sector_1510_2300.tab"
    "10884781 ../inputs/emissions/SSP5-85_region_1510_2300.tab"
    "10884779 ../inputs/emissions/SSP5-85_sector_1510_2300.tab"
    "10907190 ../profiles/output/seawater_HgT_observation_compilation.tab"
)

# Base URL for downloading
BASE_URL="https://dataverse.harvard.edu/api/access/datafile/"

# Loop through file pairs
for pair in "${file_pairs[@]}"; do
    file_id=$(echo "$pair" | awk '{print $1}')
    rel_output_path=$(echo "$pair" | awk '{print $2}')

    # Construct absolute path
    abs_output_path="${SCRIPT_DIR}/${rel_output_path}"

    # Ensure the directory exists
    mkdir -p "$(dirname "$abs_output_path")"

    # Download the file using curl
    curl -L "${BASE_URL}${file_id}" -o "${abs_output_path}" || { echo "Download failed for ${file_id}"; exit 1; }

    # convert from .tab to .csv
        python3 "${SCRIPT_DIR}/convert_tab_to_csv.py" "${abs_output_path}" --remove || { echo "Conversion failed for ${abs_output_path}"; exit 1; }
done

