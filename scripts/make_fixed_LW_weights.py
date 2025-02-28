import pandas as pd
import numpy as np
from helpers import aggregate_groups

# ----------------------------------------------------------------------------------
# Description: this script makes fixed L/W weights for `sector`. It is used to 
# generate combinations of L/W weights for the grid search sensitivity analysis.
# ----------------------------------------------------------------------------------

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_wf',  type=float, default=0.00)
parser.add_argument('--base_ws',  type=float, default=0.05)
parser.add_argument('--base_wa',  type=float, default=0.95)
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--output_directory', type=str, default='../output/sensitivity/contour_plot/')
args = parser.parse_args()

# ----------------------------------------------------------------------------------
base_wf  = args.base_wf
base_ws  = args.base_ws
base_wa  = args.base_wa
tag      = args.tag
output_directory = args.output_directory
# ----------------------------------------------------------------------------------

# --
category_dict = {
    'sector': ['Other Metals Production',
               'Gold and Silver Production', 
               'Fossil-Fuel Combustion', 
               'Mercury Production', 
               'Mercury Use'],}

# -- read template file to get time dimension
df = pd.read_csv('../inputs/emissions/LW_weights/LW_weights_sector.csv')

# -- create new dataframe with time dimension
LW_weights = df[['Year']].copy()

# -- loop over categories and set L/W weights
for cat in category_dict['sector']:
    tmp = pd.DataFrame({'Year': df['Year'].values, 
                        f'w_wf - {cat}': base_wf,
                        f'w_ws - {cat}': (1-base_wf-base_wa),
                        f'w_wa - {cat}': base_wa})
    # -- merge with existing dataframe on 'Year'
    LW_weights = pd.merge(LW_weights, tmp, on='Year', how='left')

for cat in category_dict['sector']:
    test_list = []
    for i in ['w_wf', 'w_ws', 'w_wa']:
        test_list.append(f'{i} - {cat}')     
    assert LW_weights[test_list].sum(axis=1).all() == 1, f'Weights for {cat} do not sum to 1'

LW_weights.to_csv(f'{output_directory}LW_weights_sector{tag}.csv', index=False)
