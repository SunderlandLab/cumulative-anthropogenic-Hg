import numpy as np
import pandas as pd
import json
import argparse
from helpers import get_reservoirs_and_fluxes

# --------------------------------------------------------------------------------
# Set up command line arguments and parse them
# --------------------------------------------------------------------------------

# define the parser
parser = argparse.ArgumentParser(description='Create budget tables for a given year')
parser.add_argument('--dir_path',                  type=str, help='Directory path to the output files', default='../output/main/')
parser.add_argument('--output_path',               type=str, help='Output directory path', default='../output/main/tables/')
parser.add_argument('--year',                      type=int, help='Year to construct budget for', default=2010)
parser.add_argument('--scenario',                  type=str, help='Scenario name', default='SSP1-26')
parser.add_argument('--match_dict_fn',             type=json.loads, help='Dictionary to subset the emission components driving the change. Argument should be a filepath', default=None)
parser.add_argument('--fn_label',                  type=str, help='If `match_dict` not None, this will be added to the output filenames to specify the subset of emission components', default='')
# filenames -- default filenames are constrcuted from the directory path and scenario name and year if not provided
parser.add_argument('--fn_res',                    type=str, help='Filename of the reservoirs csv file',        default=None)
parser.add_argument('--fn_rate_table',             type=str, help='Filename of the rate table csv file',        default=None)
parser.add_argument('--fn_flux_table_out',         type=str, help='Output filename for the flux table',         default=None)
parser.add_argument('--fn_flux_table_grouped_out', type=str, help='Output filename for the grouped flux table', default=None)
parser.add_argument('--fn_mass_table_out',         type=str, help='Output filename for the mass table',         default=None)
# -- 
# parse the arguments
args = parser.parse_args()

# -- directory path
dir_path      = args.dir_path
# -- year to construct budget for
year          = int(args.year)
scenario      = args.scenario

# -- dictionary to subset the emission components driving the change
match_dict    = args.match_dict_fn 
fn_label      = args.fn_label

# -- input filenames
if args.fn_res is None:
    fn_res        = dir_path+f'output_sector_{scenario}_1510_2300.csv'
else:
    fn_res        = args.fn_res
if args.fn_rate_table is None:
    fn_rate_table = dir_path+f'rate_table_sector_{scenario}_1510_2300.csv'
else:
    fn_rate_table = args.fn_rate_table
# -- output filenames
# - flux table
if args.fn_flux_table_out is None:
    fn_flux_table_out         = args.output_path+f'flux_table_sector_{scenario}_{year}.csv'
else:
    fn_flux_table_out         = args.fn_flux_table_out
# - grouped flux table
if args.fn_flux_table_grouped_out is None:
    fn_flux_table_grouped_out = args.output_path+f'flux_table_sector_{scenario}_{year}_grouped.csv'
else:
    fn_flux_table_grouped_out = args.fn_flux_table_grouped_out
# - mass table
if args.fn_mass_table_out is None:
    fn_mass_table_out         = args.output_path+f'mass_table_sector_{scenario}_{year}.csv'
else:
    fn_mass_table_out         = args.fn_mass_table_out

flux_table, flux_table_grouped, mass_table = get_reservoirs_and_fluxes(year=year, fn_reservoirs=fn_res, fn_rate_table=fn_rate_table, match_dict=match_dict)
natural_flux_table, natural_flux_table_grouped, natural_mass_table = get_reservoirs_and_fluxes(year=-2000, fn_reservoirs=fn_res, fn_rate_table=fn_rate_table, match_dict=None)

# merge 'natural mass' into 'mass_table'
natural_mass_table['natural mass [Mg]'] = natural_mass_table['mass [Mg]']
mass_table = pd.merge(mass_table, natural_mass_table[['reservoir', 'natural mass [Mg]']], on='reservoir', how='left')
mass_table['EF_alltime'] = mass_table['mass [Mg]'] / mass_table['natural mass [Mg]']

if match_dict is not None:
    fn_flux_table_out         = fn_flux_table_out.replace('.csv', f'_{fn_label}.csv')
    fn_flux_table_grouped_out = fn_flux_table_grouped_out.replace('.csv', f'_{fn_label}.csv')
    fn_mass_table_out         = fn_mass_table_out.replace('.csv', f'_{fn_label}.csv')

flux_table.to_csv(fn_flux_table_out, index=False)
flux_table_grouped.to_csv(fn_flux_table_grouped_out, index=False)
mass_table.to_csv(fn_mass_table_out, index=False)


