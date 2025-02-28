import numpy as np
import pandas as pd

import boxey as bx
# -- load data

category_dict = {'sector': ['Other Metals Production','Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production and Use'],
                 'region': ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR'],
                 'media' : ['Air', 'LW'],}

def load_boxey_emissions_data(scenario:str, category:str, separate_Hg_production_and_use:bool=False):
    '''
    Description
    -----------
    Load emissions data for Boxey model.

    Parameters
    ----------
    scenario : str
        'SSP1-26', or 'SSP5-85'
    category : str
        'sector' or 'region'
    separate_Hg_production_and_use : bool
        if True, separate mercury production and use into separate inputs
    
    Returns
    -------
    df : pd.DataFrame
        emissions data for Boxey model
    '''
    # ---- load [media, region, sector] data (1510 - 2000)
    df2 = pd.read_csv(f'../inputs/emissions/media_region_sector_1510_2000.csv')
    # - aggregate to [media, {category}] level
    for m in category_dict['media']:
        cols_media = df2.columns[df2.columns.str.contains(m)]
        for g in category_dict[category]:
            cols = cols_media[cols_media.str.contains(g)]
            df2[f'{m} - {g}'] = df2[cols].sum(axis=1)
            df2.drop(columns=cols, inplace=True)

    # -- load [media, {category}] data for a given scenario
    df3 = pd.read_csv(f'../inputs/emissions/{scenario}_media_{category}_2000_2300.csv')
    
    # - merge (1510 - 2000) and (2000 - 2300)
    df2 = df2[(df2['Year']>=1510) & (df2['Year']<=2000)].copy()
    df3 = df3[(df3['Year']>=2010) & (df3['Year']<=2300)].copy()
    # - assert columns are identical
    assert (df2.columns == df3.columns).all(), 'columns not identical'
    df = pd.concat((df2, df3), axis=0)

    # -- 
    if separate_Hg_production_and_use:
        f_production = pd.read_csv(f'../inputs/emissions/{scenario}_mercury_production_fraction_1510_2300.csv')
        drop_cols = list(f_production.columns[1:])
        df = pd.merge(df, f_production, on=['Year'], how='left')
        for c in df.columns:
            prefix = c.split(' - ')[0]
            if 'Mercury Production and Use' in c:
                df[f'{prefix} - Mercury Production'] = df[c]*df['f_production']
                df[f'{prefix} - Mercury Use'] = df[c]*(1-df['f_production'])
                drop_cols.append(c)
        df.drop(columns=drop_cols, inplace=True)

    df = df.sort_values(by='Year', ascending=True)
    df = df.reset_index(drop=True)

    # remove intermediate dataframe objects
    del df2, df3

    return df.copy()

def merge_LW_weights(df:pd.DataFrame, weights_fn:str='../inputs/emissions/LW_weights.csv'):
    ''' Merge in LW weights into emissions data. Should be called after load_boxey_emissions_data.
    
    Parameters
    ----------
    df : pd.DataFrame
        emissions data from load_boxey_emissions_data.
    
    Returns
    -------
    df : pd.DataFrame
        emissions data with LW weights merged in.
    '''
    LW_weights = pd.read_csv(weights_fn)
    df = pd.merge(df, LW_weights, on=['Year'], how='left')
    return df

# -- create inputs
def create_boxey_inputs_list(category:str, scenario:str, slice_min:int, slice_max:int, df:pd.DataFrame,
                             separate_Hg_production_and_use:bool=False):
    '''Create list of Boxey inputs for a given category, scenario, and timeslice.
    
    Parameters
    ----------
    category : str
        'sector' or 'region'
    scenario : str
        'SSP1-26', or 'SSP5-85'
    slice_min : int
        lower bound of timeslice
    slice_max : int
        upper bound of timeslice
    df : pd.DataFrame
        emissions data from load_boxey_emissions_data.
        Note that df must contain weights for LW emissions (w_wf, w_ws, w_wa)
    separate_Hg_production_and_use : bool
        if True, separate mercury production and use into separate inputs
    
    Returns
    -------
    all_inputs : list
        list of Boxey inputs
    
    '''
    all_inputs = []
    # -- 
    if separate_Hg_production_and_use:
        category_dict['sector'] = ['Other Metals Production','Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production', 'Mercury Use']

    # -- air releases
    for c in category_dict[category]:
        # -- update metadata for input
        name = f'Air_{c}_({slice_min}, {slice_max})_atm_{scenario}'
        if category == 'sector':
            sector_val = c
            region_val = None
        elif category == 'region':
            sector_val = None
            region_val = c
        # -- add input
        all_inputs.append(bx.Input(name=name, 
                                   E=(df[f'Air - {c}']).values,
                                   t=df['Year'].values, 
                                   cto='atm',
                                   meta={'name':name, 'media':'Air', 'region':region_val, 'sector':sector_val,
                                        'timeslice':[(slice_min, slice_max)], 'compartment to':'atm',
                                        'scenario':scenario}))

    # -- land and water releases
    for c in category_dict[category]:
        if category == 'sector':
            sector_val = c
            region_val = None
        elif category == 'region':
            sector_val = None
            region_val = c
        for compartment in ['wf', 'ws', 'wa']:
            # -- update metadata for input
            name = f'LW_{c}_({slice_min}, {slice_max})_{compartment}_{scenario}'
            # -- add input
            all_inputs.append(bx.Input(name=name, 
                                       E=(df[f'LW - {c}']*df[f'w_{compartment} - {c}']).values,
                                       t=df['Year'].values, 
                                       cto=compartment,
                                       meta={'name':name, 'media':'LW', 'region':region_val, 
                                             'sector':sector_val, 'timeslice':[(slice_min, slice_max)], 
                                             'compartment to':compartment, 'scenario':scenario}))
    return all_inputs

def add_pre_1510_inputs(all_inputs, scenario:str):
    df = pd.read_csv(f'../inputs/emissions/AnthroPre1510_68Gg.csv')
    all_inputs.append(bx.Input(name='Air - pre-1510 anthropogenic emissions',
                            E=df['Air'].values, t=df['Year'].values, cto='atm',
                            meta={'name': 'Air - pre-1510 anthropogenic emissions', 'media': 'Air', 'region': 'pre-1510',
                                    'sector': 'pre-1510', 'timeslice': None, 'compartment to': 'atm', 'scenario': scenario}))

    all_inputs.append(bx.Input(name='LW - pre-1510 anthropogenic emissions - ws',
                            E=(df['LW'].values*0.05),  t=df['Year'].values, cto='ws',
                            meta={'name': 'LW - pre-1510 anthropogenic emissions - ws', 'media': 'LW', 'region': 'pre-1510', 
                                    'sector': 'pre-1510', 'timeslice': None, 'compartment to': 'ws', 'scenario': scenario}))

    all_inputs.append(bx.Input(name='LW - pre-1510 anthropogenic emissions - wa',
                            E=(df['LW'].values*0.95),  t=df['Year'].values, cto='wa',
                            meta={'name': 'LW - pre-1510 anthropogenic emissions - wa', 'media': 'LW', 'region': 'pre-1510', 
                                    'sector': 'pre-1510', 'timeslice': None, 'compartment to': 'wa', 'scenario': scenario}))

    return all_inputs

def add_natural_inputs(all_inputs, E_geo_atm:int, E_geo_ocd:int, scenario:str):
    geogenic_volcanic = bx.Input(name='geogenic_volcanic',
                                 E=E_geo_atm, t=None,  cto='atm',
                                 meta={'name': 'geogenic_volcanic', 'media': 'natural', 'region': 'natural',
                                       'sector': 'natural', 'timeslice': None, 'compartment to': 'atm', 'scenario': scenario})
    all_inputs.append(geogenic_volcanic)

    geogenic_hydrothermal = bx.Input(name='geogenic_hydrothermal',
                                     E=E_geo_ocd, t=None, cto='ocd',
                                     meta={'name': 'geogenic_hydrothermal', 'media': 'natural', 'region': 'natural',
                                           'sector': 'natural', 'timeslice': None, 'compartment to': 'ocd', 'scenario': scenario})
    all_inputs.append(geogenic_hydrothermal)

    return all_inputs

# -- 
