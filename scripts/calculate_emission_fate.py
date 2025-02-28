import numpy as np
import pandas as pd
import argparse
from helpers import subset_multiple

# --------------------------------------------------------------------------------
# Define function to print the distribution of historical emissions for a given year.
# Emission subsets can be specified using the `match_dict` argument. In the command-line
# interface, the `match_dict` argument is limited to the key 'media' with values
# constructed using the `--media_values` argument.
# --------------------------------------------------------------------------------

def calculate_fate(df, match_dict={}):
    ''' Description: 
            Calculates distribution of Hg mass among compartments and burial over time
            for a given set of criteria in `match_dict`
        Example: 
            calculate_fate(df, anthro_match_dict={'media':['Air']})
    '''

    # get (annual) sum of fragments matching criteria in `match_dict``
    df_out = subset_multiple(df, match_dict=match_dict)
    df_out = df_out.groupby(by='Year', as_index=False).sum(numeric_only = True)

    # calculate cumulative burial for mass balance
    for col in ['margin_burial', 'deep_ocean_burial', 'releases']:
        df_out[col] = df_out[col].cumsum()

    # specify columns to include in output
    all_mass_cols = ['atm','tf','ts','ta','ocs','oci','ocd','wf','ws','wa', #'wi',
                     'margin_burial','deep_ocean_burial']
    
    # subset columns
    df_out = df_out[(['Year', 'releases']+all_mass_cols)]

    # calculate the total mass in system + cumulative burial
    df_out['total'] = df_out[all_mass_cols].sum(axis=1)

    return df_out

# --------------------------------------------------------------------------------
# Set up command line arguments and parse them
# --------------------------------------------------------------------------------

# define the parser
parser = argparse.ArgumentParser(description='Calculate the fate of Hg emissions')
parser.add_argument('--dir_path',     type=str, help='Path to directory containing emission data')
parser.add_argument('--scenario',     type=str, help='Scenario name', default='SSP1-26')
parser.add_argument('--year',         type=int, help='Year to calculate fate for', default=2010)
parser.add_argument('--media_values', type=str, nargs='+', help='Media values to calculate fate for')
parser.add_argument('--data_fn',    type=str, help='Input data filename. If None, default naming convention will be assumed.', default=None)

# parse the arguments
args = parser.parse_args()

dir_path      = args.dir_path     #'../output/emission_change/reference/'
scenario      = args.scenario     #'SSP1-26'
year          = args.year         #2010
media_values  = args.media_values #['Air', 'LW']

if args.data_fn is None:
    df = pd.read_csv(dir_path+f'output_sector_{scenario}_1510_2300.csv')
else:
    df = pd.read_csv(dir_path+args.data_fn)

# in the future, this could be generalized to read match_dict from json file
match_dict = {'media':media_values}

# -- call calculate_fate
df = calculate_fate(df, match_dict=match_dict)
# -- subset the data to the year
df = df[df['Year']==year]
# -- calculate cumulative releases
cum_releases = df['releases'].sum()
# -- Now print the results
print('--------------------------------')
print(f'In {year} ({scenario}), cumulative historical releases from {match_dict} : {int(cum_releases*1e-3)} Gg' )


# -- print individual compartments
print('----')
for c in ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd', 'wf', 'ws', 'wa', 'margin_burial', 'deep_ocean_burial']:
    if type(c) == str:
        value    = df[c].values.item()
        fraction = value/cum_releases
    elif type(c) == list:
        value    = df[c].sum(axis=1).values.item()
        fraction = value/cum_releases
    print(f'{c.ljust(3)} : {str(int(value*1e-3)).ljust(3)} Gg ({fraction*100:.1f}%)')
# -- print grouped compartments
print('----')
for c in [['atm'], ['tf', 'ts', 'ta'], ['ocs', 'oci', 'ocd'], ['wf', 'ws', 'wa'], ['margin_burial', 'deep_ocean_burial']]:
    if type(c) == str:
        value    = df[c].values.item()
        fraction = value/cum_releases
    elif type(c) == list:
        value    = df[c].sum(axis=1).values.item()
        fraction = value/cum_releases
    print(f'{c} : {int(value*1e-3)} Gg ({fraction*100:.1f}%)')
print('--------------------------------')
print(' ')