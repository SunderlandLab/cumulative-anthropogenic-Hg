import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--w_wf', type=float, default=None)
parser.add_argument('--w_ws', type=float, default=None)
parser.add_argument('--w_wa', type=float, default=None)
args = parser.parse_args()

path_root = '../output/sensitivity/contour_plot/'

# check if df file exists
try:
    df = pd.read_csv(f'{path_root}constraint_table_compilation_for_contour.csv')
except:
    df = pd.DataFrame(columns = ['Atmospheric Burden (Gg)', 'Preind atm. EF', 'Alltime atm. EF', 'Upper Ocean Conc. (pM)', 'Deep Ocean Conc. (pM)'])
tmp = pd.read_csv(f'{path_root}constraint_table_sector_SSP1-26_1510_2010.csv')
tmp = tmp[['Constraint Name', 'Model']].T
tmp.columns = tmp.iloc[0]
tmp = tmp.drop('Constraint Name')
tmp = tmp.reset_index(drop=True)
tmp.columns.name = None
tmp['w_wf'] = args.w_wf
tmp['w_ws'] = args.w_ws
tmp['w_wa'] = args.w_wa
df = pd.concat((df,tmp))
df.to_csv(f'{path_root}constraint_table_compilation_for_contour.csv', index=False)

# -- loop over intermediate output and remove
import os
fn_list = ['all_inputs_output_sector_SSP1-26_1510_2010.csv', 
           'budget_table_-2000_agg0_sector_SSP1-26_1510_2010.csv', 
           'budget_table_-2000_agg1_sector_SSP1-26_1510_2010.csv',
           'budget_table_2010_agg0_sector_SSP1-26_1510_2010.csv',
           'budget_table_2010_agg1_sector_SSP1-26_1510_2010.csv',
           'constraint_table_sector_SSP1-26_1510_2010.csv',
           'LW_weights_sector_contour.csv',
           'rate_table.csv']

for fn in fn_list:
    try:
        os.remove(f'{path_root}{fn}')
    except:
        pass
