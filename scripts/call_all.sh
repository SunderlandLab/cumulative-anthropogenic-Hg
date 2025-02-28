#!/bin/bash

echo "Running fetch_dataverse.sh ..."
bash fetch_dataverse.sh

echo "Running call_run_boxey.sh ..."
bash call_run_boxey.sh

echo "Running generate_waste_pool_contour_data.sh ..."
bash generate_waste_pool_contour_data.sh

echo "Running call_analysis.sh ..."
bash call_analysis.sh

echo "Completed all scripts."