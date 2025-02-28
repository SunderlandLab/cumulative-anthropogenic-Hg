import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Dict
from profiles import Datum, Profile, Collection
import json
from prepare_Hg_collection import load_collection
#from QAQC import apply_threshold

# -- load collection --
c = load_collection()

# -- add basin labels --
RECCAP = xr.open_mfdataset('../supporting_data/RECCAP2_region_masks_customized.nc')['custom_masks']
basin_mapping={0:'Land', 1:'Atlantic', 2: 'Pacific', 3: 'Indian', 4:'Arctic', 5:'Southern', 6:'Mediterranean',
               11:'Coastal Atlantic', 12:'Coastal Pacific', 13:'Coastal Indian', 14:'Coastal Arctic', 15:'Coastal Southern', 16:'Coastal Mediterranean'}
c.associate_basin_labels(ocean_basin_masks=RECCAP, mapping=basin_mapping)

# -- convert to dataframe --
df = c.to_dataframe()

# -- subset to filtered (dissolved) and unfiltered total Hg --
subset = ['Hg_T','Hg_T_D','Hg_T_D_fish',] #'Hg_T_P']
df = df[df['quantity'].isin(subset)]

# -- update the cruise IDs for the cruises presented in this work --
df.loc[df['cruiseID'] == 'A16N', 'cruiseID'] = 'A16N-2013'
df.loc[df['cruiseID'] == 'CLIVAR_I5', 'cruiseID'] = 'IO5-2009'
df.loc[df['cruiseID'] == 'CLIVAR_2011_SO', 'cruiseID'] = 'P16S-2011'

dois = {
    'KM1128':'https://doi.org/10.1002/2015GB005120', 
    'SHIPPO':'https://doi.org/10.1021/acs.est.6b04238', 
    'P16N':'https://doi.org/10.1029/2008GB003425', 
    'JC068':'https://doi.org/10.1002/2015GB005275', 
    'Cossa_2009':'https://doi.org/10.4319/lo.2009.54.3.0837', 
    'IO5-2009':'This Work',
    'GA01':'https://doi.org/10.5194/bg-15-2309-2018', 
    'GApr09':'https://doi.org/10.5285/ff46f034-f47c-05f9-e053-6c86abc0dc7e', 
    'GIPY06':'https://doi.org/10.1016/j.gca.2011.05.001', 
    'GN02':'https://doi.org/10.1038/s41598-018-32760-0', 
    'GN03':'https://doi.org/10.1038/s41598-018-32760-0', 
    'GN04':'https://doi.org/10.1016/j.marchem.2020.103855', 
    'GN05':'https://doi.org/10.1016/j.marchem.2020.103855', 
    'GA03':'https://doi.org/10.1016/j.dsr2.2014.07.004',
    'GA04N':'https://doi.org/10.1002/2017GB005700', 
    'GN01':'https://doi.org/10.1016/j.marchem.2019.103686', 
    'GP12':'https://doi.org/10.5285/ff46f034-f47c-05f9-e053-6c86abc0dc7e', 
    'GP16':'https://doi.org/10.1016/j.marchem.2016.09.005', 
    'GEOTRACES_Intercalibration':'https://doi.org/10.4319/lom.2012.10.90',
    'Kirk_2008':'https://doi.org/10.1021/es801635m', 
    'IOC':'https://doi.org/10.1016/j.marchem.2004.02.025', 
    'P16S-2011':'This Work', 
    'A16N-2013':'This Work',}

df['reference'] = df['cruiseID'].map(dois)

# drop columns that are not currently being used
df.drop(columns=['uncertainty','uncertainty_type','water_mass','month','day'], inplace=True)

# -- write to csv --
df.to_csv('../output/seawater_HgT_observation_compilation.csv', index=False)

depths_out = np.arange(0, 8000, 100, dtype=int)
c_interp = c.interpolate(depths_out=depths_out, method='linear', extrapolate=False, 
                         exclude_flags=['exceeds 10 pmol kg-1','value greater than 3 standard deviations from the profile mean']) #exclude_flags=['exceeds fixed threshold of 5 pmol kg-1'])
c_interp.associate_basin_labels(ocean_basin_masks=RECCAP, mapping=basin_mapping)
c_interp.to_dataframe().to_csv('../output/interpolated_seawater_HgT_observation_compilation.csv', index=False)