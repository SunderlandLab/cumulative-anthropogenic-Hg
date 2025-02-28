import numpy as np
import pandas as pd
import boxey as bx

groupings = {'region':['Americas', 'Europe and Former USSR', 'Africa and Middle East', 'Asia and Oceania'],
             'sector':['Fossil-Fuel Combustion', 'Gold and Silver Production', 'Mercury Production and Use', 'Other Metals Production'],
             'speciated media':['Air - Hg0', 'Air - Hg2', 'LW - HgT'],
             'media':['Air', 'LW']}

pre_1510 = {'region': {'Americas': 0, 'Europe and Former USSR': 34, 'Africa and Middle East': 0, 'Asia and Oceania': 34,},
            'sector': {'Fossil-Fuel Combustion': 0, 'Gold and Silver Production': 34, 'Mercury Production and Use': 34, 'Other Metals Production': 0,},
            'media':  {'Air': 17.5, 'LW': 50.5},}

# flatten all second level keys in pre_1510 to one level 
pre_1510_flat = {}
for k1 in pre_1510.keys():
    for k2 in pre_1510[k1].keys():
        pre_1510_flat[k2] = pre_1510[k1][k2]

# specify data types for columns
dtypes = {'Year':float, 'atm':float, 'tf':float, 'ts':float, 'ta':float, 'ocs':float, 'oci':float, 'ocd':float, 'wf':float, 'ws':float, 'wa':float, 'wi':float, 
          'margin_burial':float, 'deep_ocean_burial':float, 'name':str, 'media':str, 'region':str, 'sector':str, 'timeslice_min':float, 'timeslice_max':float, 
          'compartment to':str, 'scenario':str, 'weight':float, 'column name':str, 'releases':float}

def load_color_dict():
    color_list =   [(224,155,59), # atm
                    (75*1.3, 132*1.3, 34*1.3),(75*0.9, 132*0.9, 34*0.9), (75*0.3, 132*0.5, 34*0.5), # land
                    (58*1.5,98*1.5,155*1.5), (58,98,155), (58*0.5,98*0.5,155*0.5), # ocean
                    (255*0.6,255*0.6,255*0.6), (255*0.5,255*0.5,255*0.5), (255*0.4,255*0.4,255*0.4), (255*0.3,255*0.3,255*0.3), # waste
                    ]

    color_list = [(i[0]/255, i[1]/255, i[2]/255) for i in color_list]

    color_dict = {'atm': color_list[0], 
                  'tf': color_list[1], 'ts': color_list[2], 'ta': color_list[3], 
                  'ocs': color_list[4], 'oci': color_list[5], 'ocd': color_list[6], 
                  'wf': color_list[7], 'ws': color_list[8], 'wa': color_list[9], #'wi': color_list[10],
                  }
    return color_dict

def load_category_color_dict():
    color_dict = {}
    
    # add region colors
    for color, key in zip(['#0072BD', '#D95319', [0.7,0.7,0.7], '#EDB120'], 
                          ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR']):
        color_dict[key] = color

    palette = [[63,  63,  69 ], [42,  42,  52 ], [31,  32,  36 ],
               [123, 131, 125], [173, 168, 153], [83,  83,  79 ],
               [159, 179, 185], [195, 212, 217], [240, 249, 249],
               [144, 90,  77 ], [62,  46,  42 ],]
    palette_rgb = [np.array(p)/255 for p in palette]

    # add sector colors
    for color, key in zip([palette_rgb[3], palette_rgb[9], palette_rgb[1], palette_rgb[6]], 
                        ['Other Metals Production', 'Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production and Use']):
        color_dict[key] = color

    return color_dict

def aggregate_groups(df, groups:list, year_col=['Year']):
    ''' Description: 
            takes a DataFrame and sums columns containing substring in each element of groups.
        Example: 
            groups = ['Americas', 'Africa and Middle East'] will sum all columns containing 
            'Americas' and 'Africa and Middle East' in their name and return a DataFrame
            with columns 'Americas' and 'Africa and Middle East' '''
    for g in groups:
        cols = df.columns[df.columns.str.contains(g)].tolist()
        df[g] = df[cols].sum(axis=1)

    return df[year_col+groups]

# function to calculate cumulative emissions via interpolation of decadal emissions
def get_cumulative_emissions(df, vars:list, include_last_year=True, include_first_year=True):
    ''' Description: 
        -----------
            reads emissions and returns the cumulative emissions using linear interpolation.

        Parameters:
        -----------
            df: pd.DataFrame
                emissions DataFrame with columns ['Year', vars]
            include_last_year: bool
                include the last year in the cumulative emissions
            include_first_year: bool
                include the first year in the cumulative emissions
            vars: list
                list of variables to be summed and returned as cumulative emissions
        
        Returns:
        --------
            total_emission_Gg: pd.Series
                cumulative emissions in Gg
        
        Notes:
            - if include_last_year==False, the last year is excluded from the cumulative emissions
            - if include_first_year==False, the first year is excluded from the cumulative emissions
            - vars: list of variables to be summed and returned as cumulative emissions
        '''
    yr_min, yr_max = df['Year'].min(), df['Year'].max()
    df = pd.merge(df, pd.DataFrame({'Year':np.arange(yr_min, yr_max)}), how='outer', on='Year')
    df.sort_values(by='Year', inplace=True)
    df = df.reset_index(drop=True)
    for v in vars:
        df[v] = df[v].astype(float)
    df = df.interpolate()

    if include_last_year==False:
        df = df[df['Year']!=yr_max]
    if include_first_year==False:
        df = df[df['Year']!=yr_min]

    total_emission_Gg = df[vars].sum()*1e-3

    return total_emission_Gg

def load_800_year(category='sector', scenario='SSP1-26'):
    # aggregate over one of ['sector', 'region', 'speciated media']
    groups = groupings[category]

    if category in ['sector','region']:
        # -- pre-1510 
        df = pd.read_csv('../inputs/emissions/media_region_sector_1510_2000.csv')
        df = aggregate_groups(df, groups)
        df = df[df['Year'] <= 2000]

        # -- post-2010 (inclusive)
        tmp = pd.read_csv(f'../inputs/emissions/{scenario}_media_{category}_2000_2300.csv')
        tmp = aggregate_groups(tmp, groups)
        tmp = tmp[tmp['Year'] >= 2010]

        # -- combine
        df = pd.concat([df[['Year']+groups], tmp[['Year']+groups]], sort=False)
    
    elif category == 'speciated media':
        df = pd.read_csv(f'../inputs/emissions/{scenario}_speciated_media_region_1510_2300.csv')
        df = aggregate_groups(df, groups)

    df = df[(df['Year']>=1510) & (df['Year'] <= 2300)]

    df = df.reset_index(drop=True)

    return df


# -- move these to helpers.py
# --- Used to generate emission slices
def subset_emissions(df, vars:list, yr_min:float, yr_max:float, pad:float=1, dt:float=1):
    df = df[(df['Year'] >= yr_min) & (df['Year'] <= yr_max)].copy()
    
    start_year = -2001.
    end_year = 2300.
    zeros = list(np.repeat(0., len(vars)))
    bound_lo = pd.DataFrame([[start_year]+zeros], columns=['Year']+vars)
    bound_hi = pd.DataFrame([[end_year]+zeros],   columns=['Year']+vars)

    pad_lo = pd.DataFrame([[yr_min-pad]+zeros], columns=['Year']+vars)
    pad_hi = pd.DataFrame([[yr_max+pad]+zeros], columns=['Year']+vars)

    # add pad bounds to set step function
    df = pd.concat([pad_lo, df, pad_hi], ignore_index=True)
    # add model time bounds
    df = pd.concat([bound_lo, df, bound_hi], ignore_index=True)

    # interpolate on dt
    df['Year'] = df['Year'].astype(float)

    #df = df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='both', downcast=None)
    # create new time vector with dt
    new_t = pd.DataFrame({'Year':np.arange(start_year, end_year+dt, dt)})
    # merge new time vector with interpolated dataframe
    df = pd.merge(new_t, df, on='Year', how='outer')
    # now sort on `Year` and reset index -- important to do this before interpolation
    df.sort_values(by='Year', inplace=True)
    df = df.reset_index(drop=True)

    # interpolate on dt
    df = df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='both', downcast=None)
    
    #if (pad==1 & dt==1):
    #      df['Year'] = df['Year'].astype(int)
          
    df = df.reset_index(drop=True)
    
    return df[['Year']+vars]

from itertools import product

def make_metalist(d, weights):
     # create list of all input combinations based on the dictionary d
     iterlist = [d[k] for k in d.keys() if d[k] is not None]
     out = list(product(*iterlist))

     # ----------------------------------------------------------------------------
     # create list of dictionaries storing tags for each input
     metalist = []
     for i in out:
          # -- construct name
          name = [str(j) + '_' for j in i if j is not None] # create list of non-None values
          name = ''.join(f'{i}' for i in name) # join list of strings into single string
          name = name[:-1] # remove trailing underscore

          # -- construct dictionary of tags
          input_dict = {'name': name}
          for k,v in zip(d.keys(), i):
               if v is not None:
                    input_dict[k] = v
               else:
                    input_dict[k] = None

          # -- add weights from *SEPARATE* weights dictionary
          input_dict['weight'] = weights[input_dict['compartment to']]

          # -- add input dictionary to list of dictionaries
          metalist.append(input_dict)

     return out, metalist

def construct_inputs_from_metalist(metalist, df, pad=1, dt=1):
     ''' 
     This function constructs a list of inputs from a list of dictionaries.
     The list of dictionaries is generated from the function make_metalist.
     The inputs are constructed from the dictionary values and the dataframe df.
     
     metalist is a list of dictionaries, each of which contains the following keys:
          - name
          - sector
          - media
          - compartment to
          - weight
          - meta
     '''
     
     all_inputs = []
     # -- loop over all input dictionaries and create inputs
     for input_dict in metalist:
          #print(input_dict)
          #col_name = input_dict['sector'] + ' ' + input_dict['media']
          col_name = input_dict['column name']

          # -- load emissions and time from dataframe
          if input_dict['timeslice'] is None:
               raw_E = df[col_name].values
               t     = df['Year'].values

          # -- slice emissions if timeslice is specified
          elif input_dict['timeslice'] is not None:
               print('slicing emissions for ' + input_dict['name'])
               yr_min = input_dict['timeslice'][0]
               yr_max = input_dict['timeslice'][1]
               df_slice = subset_emissions(df, col_name, yr_min, yr_max, pad=pad, dt=dt)

               raw_E = df_slice[col_name].values
               t     = df_slice['Year'].values
          
          else:
               raise ValueError('timeslice not specified')
               
          # -- multiply emissions by weight
          weight = input_dict['weight']
          E = weight*raw_E

          # -- create input and append to list
          all_inputs.append(bx.Input(name=input_dict['name'], E=E, t=t, cto=input_dict['compartment to'], meta=input_dict))

     return all_inputs # return list of inputs

def generate_column_name(input_dict, name_order=['media', 'sector']):
     ''' 
     This function generates a column name from a dictionary of inputs.
     The column name is generated by concatenating the values of the dictionary
     in the order specified by name_order.
     '''
     col_name = [input_dict[k] + ' - ' for k in name_order if input_dict[k] is not None]
     col_name = ''.join(f'{i}' for i in col_name)
     col_name = col_name[:-3]
     return col_name

#generate_column_name(i, name_order=['media', 'sector'])

def get_expected_releases(input_dict, fn_out='../output/expected_releases.csv', save_file=False):
    # -----------------------------------------------------------------------------
    # Write inputs to csv file for debugging and mass balance checking
    # in `assess_fraction_remaining.ipynb`
    # -----------------------------------------------------------------------------
    emission_df = pd.DataFrame()

    for i in input_dict:

        # check if i.raw_E is a float or a list
        if isinstance(i.raw_E, float):
            raw_E = np.array([i.raw_E])
            raw_t = np.array([i.raw_t])
        elif isinstance(i.raw_E, int):
            raw_E = np.array([i.raw_E])
            raw_t = np.array([i.raw_t])
        else:
            raw_E = i.raw_E
            raw_t = i.raw_t

        tmp_df = pd.DataFrame({'releases': raw_E, 'Year': raw_t})
        tmp_df['name'] = i.name
        for k, v in i.meta.items():
            if (k == 'timeslice'):
                if v is not None:
                    #print(v)
                    tmp_df['timeslice_min'] = v[0][0]
                    tmp_df['timeslice_max'] = v[0][1]
                else:
                    tmp_df['timeslice_min'] = None
                    tmp_df['timeslice_max'] = None
            else:
                tmp_df[k] = v
        emission_df = pd.concat([emission_df, tmp_df], ignore_index=True)
        
    for col in emission_df.columns:
        emission_df[col] = emission_df[col].astype(dtypes[col])

    if save_file == True:
        emission_df.to_csv(fn_out, index=False)
    
    return emission_df

# function to get emission types for a given set of reservoirs -- used by `get_legacy()` below
def get_sediment_burial(model, fragments):
    
    flux_collections = {'margin_burial': ['k_L_riv_f', 'k_L_riv_s', 'k_L_riv_a', 'k_Wl_wf_riv_L', 'k_Wl_ws_riv_L', 'k_Wl_wa_riv_L'],
                        'deep_ocean_burial': ['k_Oc_sp3'],}

    reservoirs = ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd', 'wf', 'ws', 'wa', ]#'wi']
    output = pd.DataFrame({col: fragments[col] for col in fragments.columns if col not in reservoirs})
    
    for key in flux_collections.keys():
        for process in flux_collections[key]:
            # optional -- check that emission is to atmosphere
            #assert model.processes[process].compartment_to == 'atm'
            # get rate coefficient
            k = model.processes[process].get_k()
            # source compartment
            compartment_from = model.processes[process].compartment_from
            M = fragments[compartment_from].values
            E = k*M

            if key not in output.keys():
                output[key] = E
            else:
                output[key] += E
    output = pd.DataFrame(output)
    
    # ensure ['Year', margin_burial', 'deep_ocean_burial'] are the first three columns
    cols = output.columns.tolist()
    # get columns that are not 'Year', 'margin_burial', or 'deep_ocean_burial'
    other_cols = [col for col in cols if col not in ['Year', 'margin_burial', 'deep_ocean_burial']]
    # reorder columns
    output = output[['Year', 'margin_burial', 'deep_ocean_burial'] + other_cols]

    for col in output.columns:
        output[col] = output[col].astype(dtypes[col])

    return output

def get_fluxes_from_solution_object(solution_object, source_attrs, times=[2010], fn_out='./output/fluxes.csv', save_file=False):
    ''' Description:
            Get fluxes from solution object for a given set of times.
        Example:
            get_fluxes(solution_object, times=[2010, 2020])
    '''
    first = True
    for k in solution_object.keys():
        for t in times:
            row = solution_object[k].get_fluxes(time=t)
            row['Year'] = t
            # -- add metadata
            row['name'] = k
            for key in source_attrs[k].keys():
                row[key] = source_attrs[k][key]
            
            row = pd.DataFrame(row, index=[0])
            if first:
                out = row
                first = False
            else:
                out = pd.concat([out, row])

    for col in out.columns:
        if col in dtypes.keys():
            out[col] = out[col].astype(dtypes[col])
        #else:
        #    out[col] = out[col].astype(float)
    
    if save_file == True:
        out.to_csv(fn_out, index=False)

    return out

def subset_single(df:pd.DataFrame, match_column:str, match_values=None, print_match_values=False):
    ''' Description: 
            Subset a DataFrame based on a single column and a single value or list of values
        Example: 
            subset_single(df=reservoirs, match_column='name', match_values=['geogenic_volcanic', 'geogenic_hydrothermal'])
    '''
    if match_values == None:
        return df.copy()
    else:
        # if string is passed, convert to list for for loop below
        if type(match_values) == str:
            match_values = [match_values]
        # initialize empty DataFrame
        df_out = pd.DataFrame()
        for value in match_values:
            if print_match_values == True:
                print(match_column, value)
            if len(df_out) == 0:
                df_out = df[df[match_column]==value].copy()
            else:
                df_out = pd.concat((df_out, df[df[match_column]==value]))
    return df_out

def subset_multiple(df:pd.DataFrame, match_dict={}):
    ''' Description: 
            Subset a DataFrame based on a dictionary of columns and values.
            This function is a wrapper for subset_single(), which is called for each key in the dictionary.
        Example:
            subset_multiple(df=reservoirs, match_dict={'name':['geogenic_volcanic', 'geogenic_hydrothermal']})
    '''
    # initialize empty DataFrame
    df_out = df.copy()
    for key in match_dict.keys():
        df_out = subset_single(df_out, match_column=key, match_values=match_dict[key])
    return df_out

# function which accepts input criteria, subsets, and returns a dataframe with either the
# fractional distribution of that emission among compartments or the total
name   = None
media  = None # ['LW', 'Air', 'nan']
sector = None # ['Gold and Silver Production', ..., 'natural', 'nan']
region = None # ['Americas', ..., 'natural', 'nan']

match_dict = {'name': None, 'media': None, 'sector': None, 'region': None}

def groupby_year(df, match_dict):
    ''' Description: 
            Subsets df based on match_dict, groups by year, and sums all numeric columns
        Example: 
            groupby_year(burial, match_dict={'media':['Air']})
    '''
    df_out = subset_multiple(df, match_dict)
    df_out = df_out.groupby(by='Year', as_index=False).sum(numeric_only = True)
    return df_out

def add_attrs(df, source_attrs, key):
    for k in source_attrs[key]:
        if k == 'timeslice':
            df['timeslice_min'] = np.min(source_attrs[key][k])
            df['timeslice_max'] = np.max(source_attrs[key][k])
        else:
            df[k] = source_attrs[key][k]

    return df

def solution_object_to_df(solution_object, source_attrs):
    fragments = pd.DataFrame()
    for k in solution_object:
        tmp = solution_object[k].to_dataframe()
        tmp = add_attrs(tmp, source_attrs, k)
        fragments = pd.concat([fragments, tmp], ignore_index=True)

    for col in fragments.columns:
        fragments[col] = fragments[col].astype(dtypes[col])

    return fragments

# --------------------------------------------------------------------------------
# Define function to get the reservoirs and fluxes for a given year
# --------------------------------------------------------------------------------
def get_reservoirs_and_fluxes(year=2010, 
                              fn_reservoirs='../output/emission_change/reference/output_sector_SSP1-26_1510_2300.csv', 
                              fn_rate_table='../output/emission_change/reference/rate_table_sector_SSP1-26_1510_2300.csv',
                              match_dict=None):

    """ 
    This function reads the reservoirs and rate table and computes the fluxes for a given year.
    The function returns the rate table, the rate table grouped by 'Group From', 'Group To', and 'Year',
    and the total amount in each compartment for the given year.

    Parameters:
    -----------
    year: int
        The year of interest
    fn_reservoirs: str
        The filename of the reservoirs csv file
    fn_rate_table: str
        The filename of the rate table csv file
    match_dict: dict
        A dictionary that specifies the sectors and media to include in the analysis. 
        The dictionary should have the following format:
        match_dict = {'sector': ['sector1', 'sector2', ...], 'media': ['media1', 'media2', ...]}
        If match_dict is None, then all sectors and media are included in the analysis.
        Note: this is useful for getting the fluxes attributable to a specific sector or media (or combination thereof)
    
    Returns:
    --------
    flux_table: pd.DataFrame
        Table with the fluxes for the given year [subreservoirs]
    flux_table_grouped: pd.DataFrame
        Table with fluxes grouped by 'Group From', 'Group To', and 'Year' [major reservoirs]
    mass_table: pd.DataFrame
        The mass in each compartment for the given year

    Example:
    --------
    flux_table, flux_table_grouped, mass_table = get_reservoirs_and_fluxes(year=2010, fn_reservoirs=fn_res, fn_rate_table=fn_rate_table, match_dict=None)

    """
    # define the compartment names 
    comp_names = {'atm': 'Atmosphere', 
                  'tf' : 'Terrestrial Fast', 
                  'ts' : 'Terrestrial Slow', 
                  'ta' : 'Terrestrial Protected',
                  'ocs': 'Ocean Surface', 
                  'oci': 'Ocean Intermediate', 
                  'ocd': 'Ocean Deep', 
                  'wf' : 'Waste Fast', 
                  'ws' : 'Waste Slow', 
                  'wa' : 'Waste Protected', 
                  ''   : 'Sediment Burial'}
    
    # reverse keys and values in the comp_names dictionary and assign the new dictionary to comp_short
    comp_short = {v: k for k, v in comp_names.items()}

    # -- read the reservoirs and rate table
    df = pd.read_csv(fn_reservoirs)
    flux_table = pd.read_csv(fn_rate_table)
    flux_table.fillna('', inplace=True)

    # subset the dataframe to only include the year of interest
    df = df[df['Year']==year]

    # filter the dataframe to only include the sectors and media in the match_dict
    if match_dict is not None:
        # filter the dataframe to only include the sectors and media in the match_dict
        for key, value in match_dict.items():
            df = df[df[key].isin(value)]

    # sum the reservoirs (over all slices) to get the total amount in each compartment
    res = ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd', 'wf', 'ws', 'wa']
    df = df[res].sum(axis=0, numeric_only=True)

    # add 'flux' column to flux_table by multiplying the rate by the source reservoir
    for i, row in flux_table.iterrows():
        cf = row['Compartment From']
        ct = row['Compartment To']
        cf_short = comp_short[cf]
        value = row['Rate'] * df[cf_short]
        flux_table.loc[i, 'Flux'] = value #assign_significant_digits(value)

    # specify the year for the output table
    flux_table['Year'] = year

    # "super group" the compartments (e.g., all ocean compartments are grouped together)
    d = {'Atmosphere'            : 'Atmosphere',
         'Terrestrial Fast'      : 'Terrestrial', 
         'Terrestrial Slow'      : 'Terrestrial', 
         'Terrestrial Protected' : 'Terrestrial',
         'Ocean Surface'         : 'Ocean', 
         'Ocean Intermediate'    : 'Ocean', 
         'Ocean Deep'            : 'Ocean',
         'Waste Fast'            : 'Waste', 
         'Waste Slow'            : 'Waste', 
         'Waste Protected'       : 'Waste',
         ''                      : 'Sediment Burial'}

    # group every 'Compartment From' and 'Compartment To' by the dictionary d
    flux_table['Group From'] = flux_table['Compartment From'].map(d)
    flux_table['Group To']   = flux_table['Compartment To'].map(d)

    # reorder columns
    flux_table = flux_table[['Compartment From', 'Group From', 'Compartment To', 'Group To', 'Year', 'Rate', 'Flux']]

    # round the flux to the nearest integer
    flux_table['Flux'] = np.round(flux_table['Flux'],0).astype(int)

    # group the rate table by 'Group From', 'Group To', and 'Year' and sum the fluxes
    flux_table_grouped = flux_table.groupby(by=['Group From', 'Group To', 'Year'], as_index=False).sum(numeric_only=True)

    # update df
    df  = pd.DataFrame(df, columns=['mass [Mg]'])
    df.reset_index(inplace=True)
    df['reservoir'] = df['index'].map(comp_names)
    df['year'] = year
    mass_table = df[['reservoir', 'mass [Mg]', 'year']]

    return flux_table, flux_table_grouped, mass_table

# function to adjust tick and spine parameters
def modify_ticks(ax):
    c = '0.2'
    # set spine and tick color
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
    # set spine width to 0.5
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    # set tick color 
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    return ax

# function to calculate cumulative emissions via interpolation of decadal emissions
def get_cumulative_emissions(df, vars:list, include_last_year=True, include_first_year=True):
    ''' Description: reads emissions and returns the cumulative emissions using linear interpolation.
        Notes:
            - if include_last_year==False, the last year is excluded from the cumulative emissions
            - if include_first_year==False, the first year is excluded from the cumulative emissions
            - vars: list of variables to be summed and returned as cumulative emissions
        '''
    yr_min, yr_max = df['Year'].min(), df['Year'].max()
    df = pd.merge(df, pd.DataFrame({'Year':np.arange(yr_min, yr_max)}), how='outer', on='Year')
    df.sort_values(by='Year', inplace=True)
    df = df.reset_index(drop=True)
    for v in vars:
        df[v] = df[v].astype(float)
    df = df.interpolate()

    if include_last_year==False:
        df = df[df['Year']!=yr_max]
    if include_first_year==False:
        df = df[df['Year']!=yr_min]

    total_emission_Gg = df[vars].sum()*1e-3

    return total_emission_Gg