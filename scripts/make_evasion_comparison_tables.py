import numpy as np
import pandas as pd
from helpers import subset_multiple, load_800_year, get_reservoirs_and_fluxes
import argparse

def make_evasion_budget(path:str='../output/emission_change/reference/', reservoir_fn_prefix='all_inputs_output', scenario:str='SSP1-26', sel_year:int=2010, match_dict=None):
        '''
        This function calculates the evasion budget for a given year and scenario.
        The evasion budget is calculated as the sum of the following components:
        1. primary natural emissions
        2. legacy natural emissions
        3. primary anthropogenic emissions
        4. legacy anthropogenic emissions
        
        The function returns a table with the evasion budget components for the given year and scenario.

        Parameters
        ----------
        path : str
            Directory path to the output files
        reservoir_fn_prefix : str
            String to specify the prefix of the reservoirs filename. Should be one of ['all_inputs_output', 'output']. 
            When run_boxey.py is called with --Flag_Run_Sources_Separately set to True, the prefix should be 'output'.
            When run_boxey.py is called with --Flag_Run_Sources_Separately set to False, the prefix should be 'all_inputs_output'.
        scenario : str
            Scenario name. Should be one of ['SSP1-26', 'SSP5-85']
        sel_year : int
            Year to construct the evasion budget for
        match_dict : dict
            Dictionary to subset the emission components driving the change. 
            Argument should be a dictionary with a key specifying the column name and values specifying the values to match.
            This is passed to the `subset_multiple` function in the `helpers` module.
        
        Returns
        -------
        output : pd.DataFrame
            Table with the evasion budget components for the given year and scenario
        '''

        natural_year = -2000

        # get the fluxes for the natural year
        ft_natural, ft_grouped_natural, mt_natural = get_reservoirs_and_fluxes(year=natural_year, 
                fn_reservoirs=f'{path}{reservoir_fn_prefix}_sector_{scenario}_1510_2300.csv', 
                fn_rate_table=f'{path}rate_table.csv',
                match_dict=match_dict)

        # get the fluxes for the selected year
        ft_sel, ft_grouped_sel, mt_sel = get_reservoirs_and_fluxes(year=sel_year, 
                fn_reservoirs=f'{path}{reservoir_fn_prefix}_sector_{scenario}_1510_2300.csv', 
                fn_rate_table=f'{path}rate_table.csv',
                match_dict=match_dict)

        # read the primary anthropogenic emissions
        E_primary = load_800_year(scenario=scenario, category='speciated media')
        cats = ['Air - Hg0', 'Air - Hg2']
        if sel_year not in E_primary['Year'].values:
            print(f'Year {sel_year} not in the primary emissions table')
            E_primary = 0
        else:
            E_primary = E_primary[E_primary['Year'].isin([sel_year])][cats].sum(axis=1).item()

        # -- calculate the evasion budget
        
        # 1. primary natural emissions
        list_fluxes   = [230]
        list_res_from = ['None']
        list_emission_type = ['primary natural']

        # 2. legacy natural emissions
        ft_grouped_natural = subset_multiple(df=ft_grouped_natural, match_dict={'Group To':['Atmosphere']})
        list_fluxes   = list_fluxes + ft_grouped_natural['Flux'].to_list()
        list_res_from = list_res_from + ft_grouped_natural['Group From'].to_list()
        list_emission_type = list_emission_type + ['legacy natural']*len(ft_grouped_natural)

        # 3. primary anthropogenic emissions
        list_fluxes   = list_fluxes + [E_primary]
        list_res_from = list_res_from + ['None']
        list_emission_type = list_emission_type + ['primary anthropogenic']

        # 4. legacy anthropogenic emissions - calculate as the difference between the selected year and the natural year
        ft_grouped_sel     = subset_multiple(df=ft_grouped_sel,     match_dict={'Group To':['Atmosphere']})
        ft_grouped_natural = subset_multiple(df=ft_grouped_natural, match_dict={'Group To':['Atmosphere']})

        legacy_emissions = ft_grouped_sel['Flux'].values - ft_grouped_natural['Flux'].values
        list_fluxes   = list_fluxes + legacy_emissions.tolist()
        list_res_from = list_res_from + ft_grouped_sel['Group From'].to_list()
        list_emission_type = list_emission_type + ['legacy anthropogenic']*len(legacy_emissions)

        output = pd.DataFrame({'Emission Type': list_emission_type, 'Reservoir From': list_res_from, 'Flux': list_fluxes, 'Year': sel_year, 'Scenario': scenario})

        return output

# ----------------------------------------------------------------------------------------------
# Make comparison table for the evasion budget (natural, 2010, 2100 SSP1-2.6, 2100 SSP5-8.5)
# output saved to: 
#       f'{path}/tables/evasion_budget_comparison.csv'
#       f'{path}/tables/evasion_budget_comparison_grouped.csv'

if __name__ == '__main__':
    # make evasion budget
    # define the parser
    parser = argparse.ArgumentParser(description='Create budget tables for a given year')
    parser.add_argument('--dir_path', type=str, default='../output/main/', help='Directory path to the output files')
    parser.add_argument('--reservoir_fn_prefix', type=str, default='all_inputs_output', help='String to specify the prefix of the reservoirs filename. Should be one of [all_inputs_output, output]. When run_boxey.py is called with --Flag_Run_Sources_Separately set to True, the prefix should be output. When run_boxey.py is called with --Flag_Run_Sources_Separately set to False, the prefix should be all_inputs_output.')
    parser.add_argument('--output_path', type=str, default='../output/main/tables/', help='Directory path to save the output files')
    parser.add_argument('--match_dict_key', type=str, default=None, help='Dictionary to subset the emission components driving the change. Argument should be a dictionary with a key specifying the column name and values specifying the values to match. This is passed to the subset_multiple function in the helpers module.')
    parser.add_argument('--match_dict_values', type=str, nargs='+', default=None, help='Values to match in the match_dict')
    args = parser.parse_args()

    # -- unpack the arguments
    path = args.dir_path
    reservoir_fn_prefix = args.reservoir_fn_prefix
    output_path = args.output_path
    # -- set the match_dict to None; hard-coded for now
    if (args.match_dict_key is not None) and (args.match_dict_values is not None):
        match_dict = {args.match_dict_key: args.match_dict_values}
        print(f'match_dict: {match_dict}')
    else:
        match_dict = None

    i = 0
    for sel_year, scenario, column_label in zip([-2000, 2010, 2100, 2100], ['SSP1-26', 'SSP1-26', 'SSP1-26', 'SSP5-85'], ['Natural', '2010', '2100 (SSP1-2.6)', '2100 (SSP5-8.5)']):
        output = make_evasion_budget(path=path, reservoir_fn_prefix=reservoir_fn_prefix, scenario=scenario, sel_year=sel_year, match_dict=match_dict)
        if i == 0:
            comparison = output[['Emission Type', 'Reservoir From', 'Flux']].copy()
            comparison.rename(columns={'Flux': column_label}, inplace=True)
            i += 1
        else:
            comparison[column_label] = output['Flux']

    comparison_grouped = comparison.groupby(by='Emission Type', as_index=False).sum(numeric_only=True)
    # set ordering for the rows
    order = ['primary natural', 'legacy natural', 'primary anthropogenic', 'legacy anthropogenic']
    comparison_grouped['Emission Type'] = pd.Categorical(comparison_grouped['Emission Type'], categories=order, ordered=True)
    comparison_grouped.sort_values(by='Emission Type', inplace=True)

    # add total row to both comparison and comparison_grouped
    comparison.loc['Total'] = comparison.sum(numeric_only=True)
    comparison_grouped.loc['Total'] = comparison_grouped.sum(numeric_only=True)

    # save the comparison tables 
    comparison.to_csv(f'{output_path}evasion_budget_comparison.csv')
    comparison_grouped.to_csv(f'{output_path}/evasion_budget_comparison_grouped.csv')

    # display comparison and comparison_grouped using :2g format
    # set display options for pandas
    pd.options.display.float_format = '{:.4g}'.format
    print(' ---- Evasion budget comparison ---- ')
    print(comparison)
    print(' -- grouped -- ')
    print(comparison_grouped)
    print(' ----------------------------------- ')
