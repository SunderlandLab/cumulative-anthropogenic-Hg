import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict
from profiles import Datum, Profile, Collection
import json
from QAQC import apply_fixed_threshold, apply_std_outlier_detection

def update_columns(df, column_dict:Dict={}):
    """Update column names in a DataFrame using a dictionary.
       Dictionary structure is {old_name: new_name}."""
    df = df.rename(columns=column_dict)
    return df

def add_col(df, col_name:str, col_value):
    """Add a column with a constant value to a DataFrame."""
    df[col_name] = col_value
    return df

# create function to convert common dataframe format to Profile object
def prepare_dataframe(df, quantity, units='default units', column_dict={}, **kwargs):
    """Prepare a DataFrame for conversion to a Collection object."""
    # rename columns using column_dict
    #df = update_columns(df, column_dict)

    # remove rows with missing values
    df = df.dropna(subset=[quantity])
    if len(df) == 0:
        return None
    else:
        # add quantity and units as columns
        df = add_col(df, col_name='quantity', col_value=quantity)
        df = add_col(df, col_name='units', col_value=units)
        df['value'] = df[quantity]

        output_cols = ['quantity', 'depth', 'lat', 'lon', 'datetime', 'stationID', 'cruiseID', 'value', 'units', 'uncertainty', 'uncertainty_type', 'flag', 'reference']
        for col in output_cols:
            if col not in df.columns:
                df[col] = None
        df = df[output_cols]
        return df

def make_collection_from_dataframe(df):
    profiles = []
    for name, group in df.groupby(['quantity', 'lat', 'lon','stationID', 'cruiseID',]):# 'datetime', ]):
        profile = Profile.from_dataframe(group, quantity_col=group['quantity'].values[0])
        profiles.append(profile)
    return  Collection(profiles)

def fixed_unit_conversions(label):
    ''' return scale factor for converting units to common units for a given label'''
    rho_sw = 1.025 # kg L-1
    molar_mass_Hg = 200.59 # g mol-1
    mol_to_pmol = 1e12
    ng_to_g = 1e-9
    pg_to_g = 1e-12

    if label == 'ng L-1 to pmol kg-1':
        return (ng_to_g*mol_to_pmol)/(molar_mass_Hg*rho_sw) # output units: pmol kg-1

    elif label == 'pg L-1 to pmol kg-1':
        return (pg_to_g*mol_to_pmol)/(molar_mass_Hg*rho_sw) # output units: pmol kg-1

    elif label == 'pmol L-1 to pmol kg-1':
        return (1/rho_sw) # output units: pmol kg-1

    elif label == 'pmol kg-1 to pmol L-1':
        return rho_sw

    elif label == 'ng L-1 to pmol L-1':
        return (ng_to_g*mol_to_pmol)/molar_mass_Hg # output units: pmol L-1
    
    elif label == 'pg L-1 to pmol L-1':
        return (pg_to_g*mol_to_pmol)/molar_mass_Hg

    elif label == 'pmol L-1 to ng L-1':
        return molar_mass_Hg/(ng_to_g*mol_to_pmol)
    
    else:
        raise ValueError(f'Conversion not found for {label}')

def special_processing(df, cruiseID:str):
    if cruiseID == 'SHIPPO':
        pass
    elif cruiseID == 'Cossa_2009':
        v_list = ['depth', 'Hg_T',]
        df[v_list] = df[v_list].replace('-', np.nan)
    elif cruiseID == 'Kirk_2008':
        v_list = ['Hg0_T']
        df[v_list] = df[v_list].replace('ND', np.nan)
    return df

# ----------------- MAIN -----------------
def load_collection():
    # read in control file
    with open('../meta/control_file_compilation.json', 'r') as f:
        control = json.load(f)

    # create empty Collection object
    c = Collection(profiles=[])

    # loop over cruises and prepare data
    for cruise in control.keys():
        print(cruise)
        control_cruise = control[cruise]

        # get list of columns to keep -- metadata and variables
        output_columns = [col for col in control_cruise['metadata'].values()]
        output_columns.extend([col for col in control_cruise['variables'].values()])
        output_columns.extend([col for col in control_cruise['additional_columns'].keys()])

        # get load cruise data and subset to cruise of interest
        fn = control_cruise['data_path']
        file_ext = fn.split('.')[-1]
        if 'dtype_dict' in control_cruise.keys():
            dtype_dict = control_cruise['dtype_dict']
        else:
            dtype_dict = {}

        if file_ext in ['csv']:
            df = pd.read_csv(fn, dtype=dtype_dict)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(fn)
        else:
            raise ValueError(f'File extension not recognized ({file_ext})')

        # add additional columns
        for col, value in control_cruise['additional_columns'].items():
            df[col] = value

        # rename columns to standard names
        df.rename(columns=control_cruise['metadata'], inplace=True)
        df.rename(columns=control_cruise['variables'], inplace=True)

        df_cruise = df[df['cruiseID'] == cruise].copy()

        # special processing for each cruise
        df_cruise = special_processing(df=df_cruise, cruiseID=cruise)

        # keep only columns of interest
        df_cruise = df_cruise[output_columns]

        # loop over variables and prepare dataframes
        for var in control[cruise]['variables'].values():
            df_var = df_cruise.dropna(subset=[var])
            df_var = prepare_dataframe(df_var, quantity=var, units=control_cruise['units'][var], column_dict=control_cruise['metadata'])
            if df_var is not None:
                df_var['datetime'] = pd.to_datetime(df_var['datetime'])
                #df_var['reference'] = ''
                c_tmp = make_collection_from_dataframe(df_var)
                c.append_collection(collection = c_tmp)

    # -- convert 0 - 360 to -180 - 180 --
    #c.convert_lon_360_to_180()

    # -- add basin labels --
    RECCAP = xr.open_mfdataset('../supporting_data/RECCAP2_region_masks_customized.nc')['custom_masks']
    basin_mapping={0:'Land', 1:'Atlantic', 2: 'Pacific', 3: 'Indian', 4:'Arctic', 5:'Southern', 6:'Mediterranean',
                11:'Coastal Atlantic', 12:'Coastal Pacific', 13:'Coastal Indian', 14:'Coastal Arctic', 15:'Coastal Southern', 16:'Coastal Mediterranean'}
    c.associate_basin_labels(ocean_basin_masks=RECCAP, mapping=basin_mapping)

    # -- convert ng L-1 to pmol L-1 --
    scale = fixed_unit_conversions('ng L-1 to pmol L-1')
    for q in ['Hg_T', 'Hg_T_D', 'Hg0_D', 'DMHg_D', 'MeHg_D', 'MMHg_D', 'MeHg_T', 'Hg_T_P', 'MeHg_P', 'Hg_T_D_fish']:
        c.update_units(quantity=q, conversion_factor=scale, new_units='pmol L-1', old_units='ng L-1')

    # -- convert pmol L-1 to pmol kg-1 --
    # note: assumes constant seawater density of 1.025 kg L-1 
    #scale = fixed_unit_conversions('pmol L-1 to pmol kg-1')
    #for q in ['Hg_T', 'Hg_T_D', 'Hg0_D', 'DMHg_D', 'MeHg_D', 'MMHg_D', 'MeHg_T', 'Hg_T_P', 'MeHg_P', 'Hg_T_D_fish']:
    #    c.update_units(quantity=q, conversion_factor=scale, new_units='pmol kg-1', old_units='pmol L-1')

    # -- convert pmol kg-1 to pmol L-1 --
    scale = fixed_unit_conversions('pmol kg-1 to pmol L-1')
    for q in ['Hg_T', 'Hg_T_D', 'Hg0_D', 'DMHg_D', 'MeHg_D', 'MMHg_D', 'MeHg_T', 'Hg_T_P', 'MeHg_P', 'Hg_T_D_fish']:
        c.update_units(quantity=q, conversion_factor=scale, new_units='pmol L-1', old_units='pmol kg-1')
    ##################################################################################
    # Assign flags
    ##################################################################################
    # -- apply fixed threshold to data in ['Hg_T' and 'Hg_T_D'] --
    threshold = 10 # pmol kg-1
    for p in c.profiles:
        if p.quantity in ['Hg_T', 'Hg_T_D']:
            units = p.units
            p.assign_flags(fn=apply_fixed_threshold, threshold=threshold, flag=f'exceeds {int(threshold)} {units}')
    # -- apply standard deviation outlier detection to data in ['Hg_T' and 'Hg_T_D'] --
    n_std = 3
    for p in c.profiles:
        if p.quantity in ['Hg_T', 'Hg_T_D']:
            p.assign_flags(fn=apply_std_outlier_detection, n_std=n_std, flag=f'value greater than {int(n_std)} standard deviations from the profile mean')

    return c