import numpy as np
import pandas as pd
import xarray as xr

from latlontools import *

# get depth-binned seawater volume for each basin

# -- open GEBCO bathymetry and adjust lon coordinates
GEBCO = xr.open_mfdataset('../profiles/supporting_data/GEBCO_regridded_all.nc')
# convert longitude coordinate from 0-360 to -180-180
GEBCO['lon'] = (GEBCO['lon'] + 180) % 360 - 180
# sort by lon
GEBCO = GEBCO.sortby('lon')

# -- open RECCAP region masks
RECCAP = xr.open_mfdataset('../profiles/supporting_data/RECCAP2_region_masks_customized.nc')

# -- define basin mapping -- note that this is specific to the 'custom_masks' field in the above `RECCAP..._customized.nc` dataset
# this gets used later to calculate the volume of seawater in each basin and to save the results to a csv
basin_mapping={0:'Land', 1:'Atlantic', 2: 'Pacific', 3: 'Indian', 4:'Arctic', 5:'Southern', 6:'Mediterranean',
               11:'Coastal Atlantic', 12:'Coastal Pacific', 13:'Coastal Indian', 14:'Coastal Arctic', 15:'Coastal Southern', 16:'Coastal Mediterranean'}

# -- merge datasets
ds = xr.merge((GEBCO, RECCAP))

# calculate grid area [m2] based on lat/lon grid centers using latlontools
lon_b, lat_b, lon, lat = latlon_extract(ds)
area = grid_area(lon_b=lon_b, lat_b=lat_b)
area = xr.DataArray(area, coords=[lat, lon], dims=['lat', 'lon'])

ds['area'] = area
ds['area'] = ds['area'].assign_attrs({'units': 'm2'})

# make `area` a non-dimension coordinate
ds = ds.set_coords('area')

ds['volume'] = ds['area'] * (ds['elevation']*-1.)
ds['volume'] = ds['volume'].assign_attrs({'units': 'm3'})
# compare to reference volume from https://oceanservice.noaa.gov/facts/oceanwater.html
ref_vol = 1.335e9 * 1e3 * 1e3 * 1e3 # m3
assert np.abs(ds['volume'].sum().values/ref_vol) - 1 < 0.01, 'Total volume is not close to 1.335e9 km3'

# convert from m3 to L
ds['volume'] = ds['volume'] * 1e3
ds['volume'] = ds['volume'].assign_attrs({'units': 'L'})

# get volume of seawater in each basin and save to csv
output_table = {'basin':[], 'volume [L]':[]}
for basin_id in basin_mapping.keys():
    basin_name = basin_mapping[basin_id]
   # get volume of seawater in each basin
    basin_volume = ds['volume'].where(ds['custom_masks'] == basin_id).sum().values.item()
    #output_table[basin_name] = basin_volume
    output_table['basin'].append(basin_name)
    output_table['volume [L]'].append(basin_volume)

output_table = pd.DataFrame(output_table)

output_table.to_csv('../profiles/output/basin_volumes_summary.csv', index=False)


# -- get volume of seawater within each dz layer in each basin
dz = 100
depth_bins = np.arange(0, 7600, dz)
# step through each depth bin and calculate the volume of seawater in each basin within that depth bin
output_table = {'basin':[], 'depth_bin':[], 'volume [L]':[]}
# for each depth bin
for d_max in depth_bins[1:]:
    # get area for which depth (positive values) is greater than the depth bin
    ds['area_bin'] = ds['area'].where( (ds['elevation']*-1) >= d_max)

    # get volume of seawater in depth bin by multiplying area by dz
    ds['volume_bin'] = ds['area_bin'] * dz
    ds['volume_bin'] = ds['volume_bin'].assign_attrs({'units': 'm3'})
    # convert from m3 to L
    ds['volume_bin'] = ds['volume_bin'] * 1e3
    ds['volume_bin'] = ds['volume_bin'].assign_attrs({'units': 'L'})
    # step through each basin and calculate the volume of seawater in each basin within that depth bin
    for basin_id in basin_mapping.keys():
        basin_name = basin_mapping[basin_id]
        # get volume of seawater in each basin
        basin_volume = ds['volume_bin'].where(ds['custom_masks'] == basin_id).sum().values.item()
        output_table['basin'].append(basin_name)
        output_table['depth_bin'].append(f'{d_max-dz}-{d_max}')
        output_table['volume [L]'].append(basin_volume)

# now organize the output table into a dataframe where each row is a depth bin and each column is a basin
output_table = pd.DataFrame(output_table)
output_table = output_table.pivot(index='depth_bin', columns='basin', values='volume [L]')

# flatten the multi-index columns
output_table.columns = output_table.columns.get_level_values(0)

output_table['depth_min'] = [int(i.split('-')[0]) for i in output_table.index]
output_table['depth_max'] = [int(i.split('-')[1]) for i in output_table.index]

output_table = output_table.sort_values(by='depth_min')
# add 'units' column
output_table['units'] = 'L'
# reorder columns
output_table = output_table[['depth_min', 'depth_max', 'units'] + list(basin_mapping.values())]

output_table.to_csv('../profiles/output/basin_volumes_by_depth.csv', index=True)