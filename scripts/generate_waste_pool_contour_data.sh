#!/usr/bin/env bash

# OBJECTIVE
# This script generates a figure that shows the effect of varying the waste pool
# allocation of land and water releases on Hg concentrations in the atmosphere and ocean.

# DESCRIPTION
# This function iteratively performs the following steps:
# 1. Generate a LW_weights file with a unique combination of {base_wf, base_ws} values.
# 2. Run Boxey with the generated LW_weights file.
# 3. Open the `constraint_table...csv` file and add relevant constraint information
#    to a compilation file.
# 4. Delete the `constraint_table...csv` file.
# 5. Call a function to make contour plots from the compilation file.

# -- remove the compilation file if it exists -- 
# Ensure the compilation file is safely removed before starting
file="../output/sensitivity/contour_plot/constraint_table_compilation_for_contour.csv"
if [ -f "$file" ]; then
    rm "$file"
    echo "Removed existing constraint compilation file: $file"
fi

# Loop over waste factors (w_wf and w_ws)
for w_wf in $(seq 0 0.05 1.0); do
    echo "w_wf = $w_wf"
    for w_ws in $(seq 0 0.05 1.0); do
        # Skip cases where the sum exceeds 1
        sum=$(echo "$w_wf + $w_ws" | bc)
        if (( $(echo "$sum > 1" | bc -l) )); then
            continue
        else
            # Calculate remaining value for w_wa
            w_wa=$(echo "1.0 - $w_wf - $w_ws" | bc)
            
            # Round the values to three decimal places
            w_wf_in=$(printf "%.3f" "$w_wf")
            w_ws_in=$(printf "%.3f" "$w_ws")
            w_wa_in=$(printf "%.3f" "$w_wa")

            # Define the tag and file names
            tag="_contour"
            fn="../output/sensitivity/contour_plot/LW_weights_sector${tag}.csv"

            # Run the relevant Python scripts with the appropriate arguments
            python ./make_fixed_LW_weights.py --base_wf $w_wf_in --base_ws $w_ws_in --base_wa $w_wa_in --tag $tag --output_directory '../output/sensitivity/contour_plot/'
            python ./run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 2010 --Flag_Run_Pre_1510 'True' --Flag_Run_Natural 'True' --LW_weights_fn $fn --output_dir '../output/sensitivity/contour_plot/' --display_verbose 0
            python ./add_constraint_to_contour_compilation.py --w_wf $w_wf_in --w_ws $w_ws_in --w_wa $w_wa_in
        fi
    done
done

echo "Completed!"
