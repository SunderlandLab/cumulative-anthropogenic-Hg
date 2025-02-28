#!/usr/bin/env bash

# ----------------------
# -- ALL-SOURCE RUNS --
# ----------------------
echo "Running all-source runs..."
# - sector - 
# 1510 - 2300
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 2300 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1510 --slice_max 2300 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/'

# ----------------------
# -- SENSITIVITY RUNS --
# ----------------------
echo "Running sensitivity runs..."
# -- streets (2019) lower bound 
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --scale_emissions_fn '../inputs/emissions/scale_files/streets_low_sector.csv' --output_dir '../output/sensitivity/streets_low/'
# -- streets (2019) upper bound
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --scale_emissions_fn '../inputs/emissions/scale_files/streets_high_sector.csv' --output_dir '../output/sensitivity/streets_high/'
# -- low mobility
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --LW_weights_fn '../inputs/emissions/LW_weights/LW_weights_sector_low_mobility.csv' --output_dir '../output/sensitivity/low_mobility/'
# -- high mobility
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --LW_weights_fn '../inputs/emissions/LW_weights/LW_weights_sector_high_mobility.csv' --output_dir '../output/sensitivity/high_mobility/'
# -- low ocean hydrothermal input
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --E_geo_ocd 20 --output_dir '../output/sensitivity/geogenic_ocd_low/'
# -- high ocean hydrothermal input
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --E_geo_ocd 80 --output_dir '../output/sensitivity/geogenic_ocd_high/'
# -- low subaerial volcanic input
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --E_geo_atm 170 --output_dir '../output/sensitivity/geogenic_atm_low/'
# -- high subaerial volcanic input
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --E_geo_atm 336 --output_dir '../output/sensitivity/geogenic_atm_high/'
# -- respiration low
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --model_params_fn '../inputs/rate_parameters/sensitivity/model_parameters_low_respiration.json' --output_dir '../output/sensitivity/respiration_low/'
# -- respiration high
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --model_params_fn '../inputs/rate_parameters/sensitivity/model_parameters_high_respiration.json' --output_dir '../output/sensitivity/respiration_high/'
# -- Shah et al. (2021) air-sea exchange
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --model_params_fn '../inputs/rate_parameters/sensitivity/model_parameters_Shah_2021_air_sea.json' --output_dir '../output/sensitivity/Shah_2021_air_sea/'
# -- previous, smaller protected soil Hg reservoir size (Smith-Downey et al., 2010)
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --model_params_fn '../inputs/rate_parameters/sensitivity/model_parameters_Smith-Downey_2010_soil_Hg_mass.json' --output_dir '../output/sensitivity/previous_protected_soil/'
# -- previous, lower particle settling rates
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --model_params_fn '../inputs/rate_parameters/sensitivity/model_parameters_previous_ocean_Hg.json' --output_dir '../output/sensitivity/previous_ocean/'

# ----------------------
# -- ATTRIBUTION RUNS --
# ----------------------
echo "Running attribution runs..."
# - sector - 
python run_boxey.py --cat 'sector' --slice_min 1510 --slice_max 2010 --Flag_Run_Sources_Separately True --output_dir '../output/main/attribution/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 2300 --Flag_Run_Sources_Separately True --output_dir '../output/main/attribution/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1510 --slice_max 2300 --Flag_Run_Sources_Separately True --output_dir '../output/main/attribution/'

# ----------------------
# -- TIME ATTRIBUTION --
# ----------------------
echo "Running time attribution..."
# - 2010 - 
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1509 --slice_max 1510 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 1600 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1600 --slice_max 1700 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1700 --slice_max 1800 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1800 --slice_max 1900 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1900 --slice_max 2000 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 2000 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2010/'
# - 2100; SSP1-2.6 - 
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1509 --slice_max 1510 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 1600 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1600 --slice_max 1700 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1700 --slice_max 1800 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1800 --slice_max 1900 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1900 --slice_max 2000 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 2000 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 2010 --slice_max 2100 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP1-26/'
# - 2100; SSP5-8.5 -
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1509 --slice_max 1510 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1510 --slice_max 1600 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1600 --slice_max 1700 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1700 --slice_max 1800 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1800 --slice_max 1900 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1900 --slice_max 2000 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 2000 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 2000 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 2010 --slice_max 2100 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --output_dir '../output/main/time_attribution/2100/SSP5-85/'

# ------------------------------------------------
# -- TIME ATTRIBUTION FOR HISTORICAL VS. FUTURE (Fig. S6) --
# ------------------------------------------------
# -- pre-1510 and natural --
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1509 --slice_max 1510 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/time_attribution/past_and_future/SSP1-26/'
# -- pre-2010 anthropogenic --
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 1510 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --Flag_Run_Sources_Separately True --output_dir '../output/main/time_attribution/past_and_future/SSP1-26/'
# -- future anthropogenic --
python run_boxey.py --cat 'sector' --scenario 'SSP1-26' --slice_min 2010 --slice_max 2300 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --Flag_Run_Sources_Separately True --output_dir '../output/main/time_attribution/past_and_future/SSP1-26/'
# -- pre-1510 and natural --
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1509 --slice_max 1510 --Flag_Run_Pre_1510 True --Flag_Run_Natural True --output_dir '../output/main/time_attribution/past_and_future/SSP5-85/'
# -- pre-2010 anthropogenic --
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 1510 --slice_max 2010 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --Flag_Run_Sources_Separately True --output_dir '../output/main/time_attribution/past_and_future/SSP5-85/'
# -- future anthropogenic --
python run_boxey.py --cat 'sector' --scenario 'SSP5-85' --slice_min 2010 --slice_max 2300 --Flag_Run_Pre_1510 'False' --Flag_Run_Natural True --Flag_Run_Sources_Separately True --output_dir '../output/main/time_attribution/past_and_future/SSP5-85/'