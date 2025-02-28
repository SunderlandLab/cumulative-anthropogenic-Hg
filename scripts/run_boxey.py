import numpy as np
import pandas as pd

import json
import argparse

import boxey as bx
from tenbox_boxey import init_model

from helpers import subset_emissions, solution_object_to_df, get_sediment_burial, get_expected_releases, get_fluxes_from_solution_object
from setup_boxey import load_boxey_emissions_data, merge_LW_weights, create_boxey_inputs_list, add_pre_1510_inputs, add_natural_inputs # scale_boxey_emissions_data, 
from diagnostics import make_rate_table, make_budget_table_from_model

# ----------------------------------------------------------------
# set these up using arguments from the command line
# ----------------------------------------------------------------
parser = argparse.ArgumentParser(description='Run Boxey model.')
parser.add_argument('--pad', type=int, default=0.05, help='length of pad (years) to add to edges of emission slices')
parser.add_argument('--dt', type=int, default=1, help='timestep (years) for model')
parser.add_argument('--category', type=str, default='sector', help='category to run model for - options are [sector, region]')
parser.add_argument('--scenario', type=str, default='SSP1-26', help='scenario to run model for - options are [None, SSP1-26, SSP5-85]')
parser.add_argument('--slice_min', type=int, default=1510, help='lower bound of timeslice to subset emissions with')
parser.add_argument('--slice_max', type=int, default=2010, help='upper bound of timeslice to subset emissions with')
parser.add_argument('--model_params_fn', type=str, default='../inputs/rate_parameters/model_parameters.json', help='filename for model parameters-containing json file')
parser.add_argument('--LW_weights_fn', type=str, default='', help='filename for LW weights-containing csv file -- if empty, will use default LW weights file with fstring fill with category')
parser.add_argument('--Flag_Run_Pre_1510', type=str, default='True', help='flag to run pre-1510 anthropogenic emissions')
parser.add_argument('--Flag_Run_Natural', type=str, default='True', help='flag to run natural emissions')
parser.add_argument('--Flag_Run_Sources_Separately', type=str, default='False', help='flag to run sources separately -- if True, will create solution object containing partial contributions from each source; if False, will create solution object containing total contributions from all sources')
parser.add_argument('--output_dir', type=str, default='../output/main/', help='path to directory to save output to')
parser.add_argument('--scale_emissions_fn', type=str, default=None, help='filename for LW weights-containing csv file -- if None, emissions will not be scaled')
parser.add_argument('--E_geo_atm', type=float, default=232., help='geogenic release to atmosphere (Mg a-1; subaerial volcanism)')
parser.add_argument('--E_geo_ocd', type=float, default=30., help='geogenic release to deep ocean (Mg a-1; hydrothermal)')
parser.add_argument('--Flag_write_budget_table', type=str, default='True', help='flag to write budget table')
parser.add_argument('--display_verbose', type=int, default=0, help='setting to control verbosity of printed output. 0 is no output, 1 is minimal output, 2 is more output (constraint table)')
args = parser.parse_args()

# -- unpack arguments here to reduce confusion later in script
pad = args.pad
dt  = args.dt
category  = args.category # ['sector', 'region']
scenario  = args.scenario # [None, 'SSP1-26', 'SSP5-85']
slice_min = args.slice_min 
slice_max = args.slice_max
verbosity = args.display_verbose

# -- 
model_params_fn = args.model_params_fn
if (args.LW_weights_fn == None) or (args.LW_weights_fn == 'None') or (args.LW_weights_fn == ''):
    LW_weights_fn = f'../inputs/emissions/LW_weights/LW_weights_{category}.csv'
else:
    LW_weights_fn = args.LW_weights_fn

# Flags to configure whether to run pre-1510 and natural emission slices
Flag_Run_Pre_1510 = args.Flag_Run_Pre_1510
Flag_Run_Natural  = args.Flag_Run_Natural

# -- assuming slice_sel is a list containing a single tuple
if ( (type(slice_min)==int) & (type(slice_max)==int) ):
    slice_sel = [(slice_min, slice_max)]
else:
    # try to convert slice_min and slice_max to integers
    try:
        slice_min = int(slice_min)
        slice_max = int(slice_max)
        if verbosity > 0:
            print(slice_min, type(slice_min), slice_max, type(slice_max))
        slice_sel = [(slice_min, slice_max)]
    except:
        if verbosity > 0:
            print(slice_min, type(slice_min), slice_max, type(slice_max))
        raise ValueError('slice_min and slice_max must be integers')

# -- set suffix to add to output filenames
tag = f'_{category}_{scenario}_{slice_min}_{slice_max}'

output_dir = args.output_dir

# ----------------------------------------------------------------
# (1) DEFINE SOME LISTS AND CONSTANTS
# ----------------------------------------------------------------
natural_reservoirs = ['atm','tf','ts','ta','ocs','oci','ocd']
waste_reservoirs   = ['wf','ws','wa']
all_reservoirs     = natural_reservoirs + waste_reservoirs

E_geo_ocd = args.E_geo_ocd #50.
E_geo_atm = args.E_geo_atm #232.

# -- reference information
category_dict = {'sector': ['Other Metals Production','Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production', 'Mercury Use'],
                 'region': ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR'],
                 'media' : ['Air', 'LW'],}

if category == 'sector':
     sector_list, region_list = category_dict['sector'], [None]
elif category == 'region':
     sector_list, region_list = [None], category_dict['region']
name_order  = ['media', category]

# ----------------------------------------------------------------
# (2) LOAD 800 YEAR (1510 - 2300) ANTHROPOGENIC RELEASES AND WEIGHTS
# ----------------------------------------------------------------
if category == 'sector':
    df = load_boxey_emissions_data(scenario=scenario, category=category, separate_Hg_production_and_use=True)
    #print(df.columns)
elif category == 'region':
    if verbosity > 0:
        print('Warning -- not distinguishing between Hg production and use for region-based emissions')
    df = load_boxey_emissions_data(scenario=scenario, category=category, separate_Hg_production_and_use=False)
elif category == 'Amos 2013':
    if verbosity > 0:
        print('Warning -- neither sector nor region selected')
    #df = pd.read_csv('/Users/bengeyman/Downloads/AnthroEmissAllTime_20120112.csv')
else:
    raise ValueError('category must be either "sector" or "region"')

# - scale emissions if file is specified
if args.scale_emissions_fn != None:
    scale_df = pd.read_csv(args.scale_emissions_fn) # read in scale factors from scale_emissions_fn
    scale_df = scale_df[df.columns] # make sure the columns in scale_df are the same as in df
    # make sure the years in scale_df are the same as in df
    assert (scale_df['Year'] == df['Year']).all(), 'Years in scale_df do not match years in df'
    # make sure the columns in scale_df are the same as in df
    assert (scale_df.columns == df.columns).all(), 'Columns in scale_df do not match columns in df'

    # loop over each column in scale_df and scale the corresponding column in df
    for col in scale_df.columns:
        if col == 'Year':
            continue
        df[col] = df[col] * scale_df[col]

if category in ['sector', 'region']:
    df = merge_LW_weights(df, weights_fn=LW_weights_fn)
else:
    print('Warning -- not merging LW weights into emissions data because category is not sector or region')

# -----------------------------------------------------------
# Slice anthropogenic inputs
# -----------------------------------------------------------
cols = df.columns
#print(cols)
cols = list(cols.drop('Year'))
df = subset_emissions(df, vars=cols, yr_min=slice_min, yr_max=slice_max, pad=pad, dt=dt)
# check for duplicate values in the 'Year' column and warn if 'verbosity' is greater than 0
if verbosity > 0:
    if df['Year'].duplicated().any():
        print('Warning -- duplicate years in emissions data')
# drop duplicate values in the 'Year' column (keep the first occurrence)
df = df.drop_duplicates(subset='Year', keep='first')

# -----------------------------------------------------------
# Create inputs
# -----------------------------------------------------------
if category == 'sector':
    all_inputs = create_boxey_inputs_list(category=category, scenario=scenario, slice_min=slice_min, slice_max=slice_max, df=df, separate_Hg_production_and_use=True)
elif category == 'region':
    all_inputs = create_boxey_inputs_list(category=category, scenario=scenario, slice_min=slice_min, slice_max=slice_max, df=df, separate_Hg_production_and_use=False)
elif category == 'Amos 2013':
    if verbosity > 0:
        print('Warning -- using custom inputs')
    all_inputs = [bx.Input(name='Amos 2013', E=(df[f'Air - central']).values, t=df['Year'].values, cto='atm',
                           meta={'name':'Amos (2013)', 'media':'Air', 'region':None, 'sector':None,
                                 'timeslice':[(slice_min, slice_max)], 'compartment to':'atm', 'scenario':None})]
else:
    raise ValueError('category must be either "sector" or "region"')

# - add pre-1510 anthropogenic inputs if specified
if Flag_Run_Pre_1510 in [True, 'True', 'true', 'TRUE']:
    if verbosity > 0:
        print('adding pre-1510 to inputs')
    all_inputs = add_pre_1510_inputs(all_inputs, scenario=scenario)
elif Flag_Run_Pre_1510 in [False, 'False', 'false', 'FALSE']:
    if verbosity > 0:
        print('not adding pre-1510 to inputs')
else:
    raise ValueError('Flag_Run_Pre_1510 must be True or False')

# - add geogenic inputs if specified
if Flag_Run_Natural in [True, 'True', 'true', 'TRUE']:
    if verbosity > 0:
        print('adding natural inputs to inputs')
    all_inputs = add_natural_inputs(all_inputs, E_geo_atm=E_geo_atm, E_geo_ocd=E_geo_ocd, scenario=scenario)
elif Flag_Run_Natural in [False, 'False', 'false', 'FALSE']:
    if verbosity > 0:
        print('not adding natural inputs to inputs')
else:
    raise ValueError('Flag_Run_Natural must be True or False')

# -----------------------------------------------------------------------------
# (3) INITIALIZE MODEL
# -----------------------------------------------------------------------------
model_time = np.arange(-2000, 2300+dt, dt) # start, end, timestep

# -- load model parameters from json
with open(model_params_fn, 'r') as f:
    model_parameters = json.load(f)

# initialize model
model = init_model(input_dict=model_parameters)

# temporary: save rate matrix to .csv
rate_matrix = pd.DataFrame(model.matrix, columns=all_reservoirs)
rate_matrix.to_csv(f'{output_dir}/rate_matrix{tag}.csv', index=False)

# add inputs to model
model = bx.add_inputs(model, all_inputs)

# -----------------------------------------------------------------------------
# (4) RUN MODEL
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (4a) run model for all inputs and save fragments
# -----------------------------------------------------------------------------
if (args.Flag_Run_Sources_Separately == True) or (args.Flag_Run_Sources_Separately in ['True', 'true', 'TRUE']):
    if verbosity > 0:
        print(args.Flag_Run_Sources_Separately)
        print('running sources separately')
    # -- consider getting value from config file instead of model_parameters
    # -- steady state contribution from subaerial volcanism
    steady_atm = model.get_steady_state(input_vector=[E_geo_atm, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # -- steady state contribution from submarine volcanism
    steady_ocd = model.get_steady_state(input_vector=[0, 0, 0, 0, 0, 0, E_geo_ocd, 0, 0, 0])

    # create dictionary of source attributes
    # -- keys are the names of the sources (used for model.run_sources())
    # -- values are dictionaries containing metadata for use in datacube
    source_attrs = {}
    initial_conditions_dict = {} # stores initial conditions for each source -- passed to model.run_sources()
    for i in all_inputs:
        # add metadata to source_attrs
        source_attrs[i.name] = i.meta

        # set initial conditions for all sources to None except for geogenic sources
        if i.name not in ['geogenic_volcanic', 'geogenic_hydrothermal']:
            initial_conditions_dict[i.name] = None
        elif i.name == 'geogenic_volcanic':
            initial_conditions_dict[i.name] = steady_atm
        elif i.name == 'geogenic_hydrothermal':
            initial_conditions_dict[i.name] = steady_ocd

    # --- The "initial_conditions_dict" is not doing what I want it to?
    solution= model.run_sources(list(source_attrs.keys()), times=model_time, initial_conditions=initial_conditions_dict)

    # -----------------------------------------------------------------------------
    # process and save solution
    # -----------------------------------------------------------------------------
    
    # -- `fragments` is a DataFrame containing the mass in each reservoir from a given (source, timeslice) input
    fragments = solution_object_to_df(solution, source_attrs)
    fragments.to_csv(f'{output_dir}/fragments{tag}.csv', index=False) # save fragments to csv
    # check for negative values
    tol = -1e-4
    assert (fragments[all_reservoirs] >= tol).all().all(), 'negative values in reservoirs'

    # -- calculate sediment burial
    sediment_burial = get_sediment_burial(model, fragments)
    #sediment_burial.to_csv(f'./output/sediment_burial{tag}.csv', index=False)

    releases = get_expected_releases(all_inputs, save_file=False)
    merge_cols = ['Year', 'name', 'media', 'region', 'sector', 'timeslice_min', 'timeslice_max', 'compartment to', 'scenario', ]#'weight', 'column name',]
    df = pd.merge(fragments, sediment_burial, on=merge_cols, how='left')
    df = pd.merge(df, releases, on=merge_cols, how='left')

    for key, value in ({'geogenic_volcanic':E_geo_atm,'geogenic_hydrothermal':E_geo_ocd}).items():
        df.loc[df['name'] == key, 'releases'] = value
    df.to_csv(f'{output_dir}/output{tag}.csv', index=False)

# -----------------------------------------------------------------------------
# (4b) run model for all inputs and save solution
# -----------------------------------------------------------------------------
elif (args.Flag_Run_Sources_Separately == False) or (args.Flag_Run_Sources_Separately in ['False', 'false', 'FALSE']):
    if verbosity > 0:
        print('running sources together')

    steady = model.get_steady_state(input_vector=[E_geo_atm, 0, 0, 0, 0, 0, E_geo_ocd, 0, 0, 0])
    solution = model.run(times=model_time, initial_conditions=steady)

    df = solution.to_dataframe()
    df.to_csv(f'{output_dir}/all_inputs_output{tag}.csv', index=False)

    # -- make budget tables if specified by flag
    if (args.Flag_write_budget_table == True) or (args.Flag_write_budget_table in ['True', 'true', 'TRUE']):
        if verbosity > 0:
            print('writing budget tables')
        years = [-2000, 2010]
        if slice_max >= 2100:
            years.append(2100)
        if slice_max >= 2300:
            years.append(2300)
        for year in years:
            for aggregation_level in [0, 1]:
                budget_table = make_budget_table_from_model(model, solution, year=year, aggregation_level=aggregation_level)
                # -- reduce precision for output
                budget_table['source mass [Mg]'] = budget_table['source mass [Mg]'].astype(int)
                budget_table['flux [Mg/yr]'] = budget_table['flux [Mg/yr]'].round(1)
                budget_table.to_csv(f'{output_dir}/budget_table_{year}_agg{aggregation_level}{tag}.csv', index=False)

    # -----------------------------------------------------------------------------
    # (5) POST-PROCESS SOLUTION
    # -----------------------------------------------------------------------------
    from diagnostics import make_constraint_table
    if verbosity > 1:
        flag_display_output = True
    else:
        flag_display_output = False

    table = make_constraint_table(model, solution, natural_emission_vector=[E_geo_atm,0,0,0,0,0,E_geo_ocd,0,0,0], oci_depth_km=1.5, display_output=flag_display_output)
    table.to_csv(f'{output_dir}/constraint_table{tag}.csv', index=False)

# -- write rate table no matter what
rate_table = make_rate_table(model)
rate_table.to_csv(f'{output_dir}/rate_table.csv', index=False)