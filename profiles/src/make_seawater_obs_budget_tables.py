import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------
# NOTES: 
# - change the path to the input files if necessary
# - current limitations:
#   - on L84 we set `volumes['depth'] = volumes['depth_max']`, which is not ideal. Need to align depth centers in 
#     interpolated profiles and basin volumes... 
 
# ------------------------------------------------------------------------------------------------------------------
# Define file paths
fn_interpolated_concentrations = '../profiles/output/interpolated_seawater_HgT_observation_compilation.csv'
fn_basin_volumes = '../profiles/output/basin_volumes_by_depth.csv'

# ------------------------------------------------------------------------------------------------------------------
# Define function to make a table of total mercury mass and concentration in the global ocean and by basin
# ------------------------------------------------------------------------------------------------------------------

def make_budget_table(df, volumes, quantity_keys=['Hg_T', 'Hg_T_D', 'Hg_T_D_fish'], groupby_keys=['basin', 'depth'], ):

    '''
    Function to make a table of total mercury mass and concentration in the global ocean and by basin.
    The table includes volume-weighted medians with 5th and 95th percentiles in brackets for concentrations and masses.
    The table also includes the number of observations in each bin.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with columns 'basin', 'depth', 'quantity', 'value', 'units'
    volumes : pandas DataFrame
        DataFrame with columns 'depth_bin', 'depth_min', 'depth_max', 'units', 'basin', 'volume [L]'
    quantity_keys : list
        List of quantity keys to include in the table
    groupby_keys : list
        List of keys to group by 

    Returns 
    -------
    totals : pandas DataFrame
        DataFrame with columns 'basin', 'volume [L]', 'conc_p50 [pM]', 'conc_p5 [pM]', 'conc_p95 [pM]', 'mass_p50 [Gg]', 'mass_p5 [Gg]', 'mass_p95 [Gg]', 'count'   
    display_table : pandas DataFrame
        DataFrame with columns 'basin', 'volume [L]', 'conc_p50 [pM]', 'mass_p50 [Gg]', 'count' 
        The display_table is formatted for display in a jupyter notebook.
    '''
    # subset the dataframe to only include the quantity_keys
    df = df[df['quantity'].isin(quantity_keys)]

    # define dictionary of summary statistics to calculate
    lambda_dict = {'min'  : lambda x: np.min(x),
                   'p5'   : lambda x: np.percentile(x, 5),
                   'p25'  : lambda x: np.percentile(x, 25),
                   'p50'  : lambda x: np.percentile(x, 50),
                   'p75'  : lambda x: np.percentile(x, 75),
                   'p95'  : lambda x: np.percentile(x, 95),
                   'max'  : lambda x: np.max(x),
                   'mean' : lambda x: np.mean(x),
                   'std'  : lambda x: np.std(x),
                   'count': lambda x: len(x),}

    # apply the lambda functions in lambda_dict to the value column and rename the column to the key in lambda_dict
    conc = df.groupby(groupby_keys, as_index=False).agg({'value': [lambda_dict[key] for key in lambda_dict.keys()]})
    # now flatten the multi-index columns and rename them using the keys in lambda_dict
    conc.columns = ['_'.join(col).strip() for col in conc.columns.values]
    # remove underline from the end of the column names
    conc.columns = [col[:-1] if col.endswith('_') else col for col in conc.columns]

    col_replacement_dict = {}
    n_keys = len(lambda_dict.keys())
    if n_keys == 1:
        col_replacement_dict[f'value_<lambda>'] = list(lambda_dict.keys())[0]
    else:
        for i in range(n_keys):
            col_replacement_dict[f'value_<lambda_{i}>'] = list(lambda_dict.keys())[i]

    conc.rename(columns=col_replacement_dict, inplace=True)

    conc['concentration units'] = 'pmol L-1'

    # reshape volumes to have column 'volume [L]' and make basin column a row value instead
    volumes = volumes.melt(id_vars=['depth_bin','depth_min','depth_max','units'], var_name='basin', value_name='volume [L]')
    volumes['depth'] = volumes['depth_max']

    # merge the concentrations with the volumes
    conc = conc.merge(volumes, on=['basin', 'depth'], how='outer')

    # add a column for volumes where Hg measurements are available
    # -- this is used as the denominator to calculate the volume-weighted concentrations
    conc['volume [L] - Hg measurements'] = conc['volume [L]'].where(~conc['count'].isnull(), 0)

    molar_mass_Hg = 200.59 # g/mol
    pmol_to_mol = 1e-12 # mol/pmol
    g_to_Gg = 1e-9

    # calculate the total mass of mercury in each bin
    for c in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'std']:
        conc[f'mass_{c} [Gg] - Hg measurements'] = conc[c] * conc['volume [L] - Hg measurements'] * pmol_to_mol * molar_mass_Hg * g_to_Gg

    agg_dict = {f'mass_{c} [Gg] - Hg measurements': 'sum' for c in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'std']}
    agg_dict['count'] = 'sum'
    agg_dict['volume [L]'] = 'sum'
    agg_dict['volume [L] - Hg measurements'] = 'sum'

    totals = conc.groupby(['basin'], as_index=False).agg(agg_dict).reset_index(drop=True)

    # add row for global total
    totals.loc[totals.shape[0]] = ['Global']+ [totals[f'mass_{c} [Gg] - Hg measurements'].sum() for c in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'std']] + [totals['count'].sum(), totals['volume [L]'].sum(), totals['volume [L] - Hg measurements'].sum()]

    # get volume-weighted concentrations in pmol L-1
    for c in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'std']:
        # -- divide the total mass by the total volume where Hg measurements are available
        totals[f'conc_{c} [pM]'] = totals[f'mass_{c} [Gg] - Hg measurements'] / (totals['volume [L] - Hg measurements'] * pmol_to_mol * molar_mass_Hg * g_to_Gg)
    
    # now recalculate mass using the volume-weighted concentrations and the total volume
    for c in ['min', 'p5', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'std']:
        totals[f'mass_{c} [Gg]'] = totals[f'conc_{c} [pM]'] * totals['volume [L]'] * pmol_to_mol * molar_mass_Hg * g_to_Gg
        
    row_order = ['Arctic',  'Coastal Arctic', 'Atlantic', 'Coastal Atlantic', 'Indian', 'Coastal Indian', 'Pacific', 'Coastal Pacific', 'Southern', 'Coastal Southern',  'Coastal Mediterranean', 'Land', 'Global']
    # sort the rows by the order in row_order
    totals = totals.set_index('basin').reindex(row_order).reset_index()

    display_cols = ['basin', 'volume [L]', 'conc_p50 [pM]', 'conc_p5 [pM]', 'conc_p95 [pM]', 'mass_p50 [Gg]', 'mass_p5 [Gg]', 'mass_p95 [Gg]', 'count']

    # display the totals with 2 decimal places precision except for the count column
    totals['count'] = totals['count'].astype(int)
    #with pd.option_context('display.float_format', '{:.3g}'.format):
    #    display(totals[display_cols])

    display_table = totals[display_cols].copy()
    # now create a string for each median [Q1, Q3] for each basin 
    for basin in display_table['basin']:

        display_table.loc[display_table['basin'] == basin, 'conc_p50 [pM]'] = f"{display_table.loc[display_table['basin'] == basin, 'conc_p50 [pM]'].values[0]:.2g} [{display_table.loc[display_table['basin'] == basin, 'conc_p5 [pM]'].values[0]:.2g}, {display_table.loc[display_table['basin'] == basin, 'conc_p95 [pM]'].values[0]:.2g}]"
        display_table.loc[display_table['basin'] == basin, 'mass_p50 [Gg]'] = f"{display_table.loc[display_table['basin'] == basin, 'mass_p50 [Gg]'].values[0]:.2g} [{display_table.loc[display_table['basin'] == basin, 'mass_p5 [Gg]'].values[0]:.2g}, {display_table.loc[display_table['basin'] == basin, 'mass_p95 [Gg]'].values[0]:.2g}]"
        
    # drop the Q1 and Q3 columns
    display_table.drop(columns=['conc_p5 [pM]', 'conc_p95 [pM]', 'mass_p5 [Gg]', 'mass_p95 [Gg]'], inplace=True)

    # don't show ['count'] column because the count includes interpolated values and is not a true count of observations
    display_table.drop(columns=['count'], inplace=True)
    
    return totals, display_table


# ------------------------------------------------------------------------------------------------------------------
# make budget table for global ocean and by basin (all depths)
# -- 
df = pd.read_csv(fn_interpolated_concentrations) # read interpolated profiles 
vols = pd.read_csv(fn_basin_volumes) # read in basin volumes

totals, display_table = make_budget_table(df, volumes=vols, quantity_keys=['Hg_T', 'Hg_T_D', 'Hg_T_D_fish'], groupby_keys=['basin', 'depth'])
print('Table S3: Total mercury mass and concentration in the global ocean and by basin.') 
print('Concentrations are volume-weighted medians with 5th and 95th percentiles in brackets.') 
print('Masses are volume-weighted medians with 5th and 95th percentiles in brackets.')
print('Counts are the number of observations in each bin.')
with pd.option_context('display.float_format', '{:.2g}'.format):
    print(display_table)

display_table['Depths'] = 'All depths'

display_table.to_csv('../profiles/output/budget_table_all_depths.csv', index=False)

print('Displaying the global weighted concentration quantiles for the whole ocean:')
for c in ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'std']:
    print(f'Global {c}: {totals[f"conc_{c} [pM]"].values[-1]:.2g} pM')


# ------------------------------------------------------------------------------------------------------------------
# make budget table for global ocean and by basin (upper 1500 m)
# -- 
df = pd.read_csv(fn_interpolated_concentrations) # read interpolated profiles 
vols = pd.read_csv(fn_basin_volumes) # read in basin volumes

df = df[df['depth'] <= 1500]
vols = vols[vols['depth_max'] <= 1500]

totals, display_table = make_budget_table(df, volumes=vols, quantity_keys=['Hg_T', 'Hg_T_D', 'Hg_T_D_fish'], groupby_keys=['basin', 'depth'])
print('Table S3.1: Total mercury mass and concentration in the upper ocean (0 - 1500 m).') 
print('Concentrations are volume-weighted medians with 5th and 95th percentiles in brackets.') 
print('Masses are volume-weighted medians with 5th and 95th percentiles in brackets.')
print('Counts are the number of observations in each bin.')
with pd.option_context('display.float_format', '{:.2g}'.format):
    print(display_table)

display_table['Depths'] = '0 - 1500m'

display_table.to_csv('../profiles/output/budget_table_above_1500m.csv', index=False)

quantile_table = {'Depth':['0 - 1500m'], 'p5':[], 'p25':[], 'p50':[], 'p75':[], 'p95':[]}

print('Displaying the global weighted concentration quantiles for the upper ocean (0 - 1500 m):')
for c in ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'std']:
    print(f'Global {c}: {totals[f"conc_{c} [pM]"].values[-1]:.2g} pM')
    if c in ['p5', 'p25', 'p50', 'p75', 'p95']:
        quantile_table[c].append(totals[f"conc_{c} [pM]"].values[-1])
# ------------------------------------------------------------------------------------------------------------------
# make budget table for global ocean and by basin (lower 1500 m)
# -- 
df = pd.read_csv(fn_interpolated_concentrations) # read interpolated profiles 
vols = pd.read_csv(fn_basin_volumes) # read in basin volumes

df = df[df['depth'] > 1500]
vols = vols[vols['depth_max'] > 1500]

totals, display_table = make_budget_table(df, volumes=vols, quantity_keys=['Hg_T', 'Hg_T_D', 'Hg_T_D_fish'], groupby_keys=['basin', 'depth'])
print('Table S3.2: Total mercury mass and concentration in the deep ocean (>1500 m).') 
print('Concentrations are volume-weighted medians with 5th and 95th percentiles in brackets.') 
print('Masses are volume-weighted medians with 5th and 95th percentiles in brackets.')
print('Counts are the number of observations in each bin.')
with pd.option_context('display.float_format', '{:.2g}'.format):
    print(display_table)

display_table['Depths'] = '>1500m'

display_table.to_csv('../profiles/output/budget_table_below_1500m.csv', index=False)

quantile_table['Depth'].append('>1500m')
print('Displaying the global weighted concentration quantiles for the deep ocean (>1500 m):')
for c in ['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'std']:
    print(f'Global {c}: {totals[f"conc_{c} [pM]"].values[-1]:.2g} pM')
    if c in ['p5', 'p25', 'p50', 'p75', 'p95']:
        quantile_table[c].append(totals[f"conc_{c} [pM]"].values[-1])

quantile_table = pd.DataFrame(quantile_table)
quantile_table = quantile_table.round(4)
quantile_table.to_csv('../profiles/output/seawater_concentration_quantiles.csv', index=False)
