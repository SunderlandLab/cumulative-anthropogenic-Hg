#!/bin/bash

# --------------------------------------
# --- make tables and display output ---
# --------------------------------------

# budget tables - reference scenario
python make_budget_tables.py --year '-2000' --scenario 'SSP1-26' --fn_res '../output/main/all_inputs_output_sector_SSP1-26_1510_2300.csv' --fn_rate_table '../output/main/rate_table.csv' --output_path '../output/main/tables/' 
python make_budget_tables.py --year 2010 --scenario 'SSP1-26' --fn_res '../output/main/all_inputs_output_sector_SSP1-26_1510_2300.csv' --fn_rate_table '../output/main/rate_table.csv' --output_path '../output/main/tables/' 
python make_budget_tables.py --year 2100 --scenario 'SSP1-26' --fn_res '../output/main/all_inputs_output_sector_SSP1-26_1510_2300.csv' --fn_rate_table '../output/main/rate_table.csv' --output_path '../output/main/tables/'
python make_budget_tables.py --year 2100 --scenario 'SSP5-85' --fn_res '../output/main/all_inputs_output_sector_SSP5-85_1510_2300.csv' --fn_rate_table '../output/main/rate_table.csv' --output_path '../output/main/tables/'
# budget tables - 80% CI for historical emissions from Streets et al. (2019)
python make_budget_tables.py --year 2010 --scenario 'SSP1-26' --fn_res '../output/sensitivity/streets_low/all_inputs_output_sector_SSP1-26_1510_2010.csv' --fn_rate_table '../output/sensitivity/streets_low/rate_table.csv' --output_path '../output/sensitivity/streets_low/tables/'
python make_budget_tables.py --year 2010 --scenario 'SSP1-26' --fn_res '../output/sensitivity/streets_high/all_inputs_output_sector_SSP1-26_1510_2010.csv' --fn_rate_table '../output/sensitivity/streets_high/rate_table.csv' --output_path '../output/sensitivity/streets_high/tables/'

# evasion budget for [natural, 2010, 2100 (1-2.6), 2100 (5-8.5)]
python make_evasion_comparison_tables.py --dir_path '../output/main/' --reservoir_fn_prefix 'all_inputs_output' --output_path '../output/main/tables/'
# evasion attributable to L/W emissions only
python make_evasion_comparison_tables.py --dir_path '../output/main/attribution/' --reservoir_fn_prefix 'output' --output_path '../output/main/attribution/tables/LW_only/' --match_dict_key 'media' --match_dict_values 'LW'

# output distribution (fate) of historical emissions for a given year (2010 is base)
python calculate_emission_fate.py --dir_path '../output/main/attribution/' --year 2010 --scenario 'SSP1-26' --media_values 'Air' 'LW'
python calculate_emission_fate.py --dir_path '../output/main/attribution/' --year 2010 --scenario 'SSP1-26' --media_values 'Air'
python calculate_emission_fate.py --dir_path '../output/main/attribution/' --year 2010 --scenario 'SSP1-26' --media_values 'LW'

# --------------------------------------
# --- profiles ---
# --------------------------------------
python ../profiles/src/misc_scripts/calculate_basin_volumes.py
python ../profiles/src/make_seawater_obs_budget_tables.py

# ----------------------
# --- plot figures   ---
# ----------------------
python ./plot_emissions.py
echo "... saved figure 1 ..."

python ./plot_observational_constraint_evaluation.py
echo "... saved figure 2 ..."

# figure 3 made in illustrator using data from 
# ../output/main/mass_table_sector_SSP1-26_2010.csv and ../output/main/flux_table_sector_SSP1-26_2010.csv

python ./plot_emission_attribution.py
echo "... saved figure 4 ..."

python ../profiles/src/misc_scripts/plot_station_map.py
echo "... saved figure S1 ..."

python ../profiles/src/plot_basin_depth_profiles.py
echo "... saved figure S2 ..."

python ./plot_emission_region_map.py
echo "... saved figure S3 ..."

python ./plot_historical_inventory_comparison.py
echo "... saved figure S4 ..."

python ./plot_waste_contour_figure.py
echo "... saved figure S5 ..."

python ./plot_historical_future_reservoir_contributions.py
echo "... saved figure S6 ..."

python ./plot_ocean_attribution_pie_chart.py
echo "... saved figure S8 ..."

python ./plot_EF_vs_lifetime_comparison.py
echo "... saved figure S9 ..."

python ./write_tables.py
echo "... saved tables ..."

# ----------------------------------
# --- remove intermediate output ---
# -----------------------------------
#python cleanup.py
#echo "... removed intermediate output ..."