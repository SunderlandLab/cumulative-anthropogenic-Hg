import xarray as xr

# associate basin labels with each profile in the collection
RECCAP = xr.open_mfdataset('../../supporting_data/RECCAP2_region_masks_all_v20221025.nc')

# convert longitude coordinate from 0-360 to -180-180
RECCAP['lon'] = (RECCAP['lon'] + 180) % 360 - 180
# sort by lon
RECCAP = RECCAP.sortby('lon')

# -- make custom masks based on RECCAP 
# initialize custom mask as all-land
RECCAP['custom_masks'] = (RECCAP['open_ocean']*0)

# add mediterranean mask (6)
RECCAP['mediterranean'] = RECCAP['atlantic'].where(RECCAP['atlantic'] == 6, 0).copy()

# assign open ocean basins (1. Atlantic, 2. Pacific, 3. Indian, 4. Arctic, 5. Southern, 6. Mediterranean)
for basin_name, basin_val in zip(['atlantic', 'pacific', 'indian', 'arctic', 'southern', 'mediterranean'], [1, 2, 3, 4, 5, 6]):
    RECCAP['custom_masks'] = RECCAP['custom_masks'].where(RECCAP[basin_name] == 0, basin_val)

# make coast mask 0 (non-coastal) or 1 (coastal)
RECCAP['coast'] = RECCAP['coast'].where(RECCAP['coast'] == 0, 10)

# assign coastal basins (regular basin ids + 10)
RECCAP['custom_masks'] = RECCAP['custom_masks'] + RECCAP['coast']

# add attributes
RECCAP['custom_masks'] = RECCAP['custom_masks'].assign_attrs({'open_ocean_labels': '0. Land, 1. Atlantic, 2. Pacific, 3. Indian, 4. Arctic, 5. Southern, 6. Mediterranean',
                                                              'coastal_labels': '11. Coastal Atlantic, 12. Coastal Pacific, 13. Coastal Indian, 14. Coastal Arctic, 15. Coastal Southern, 16. Coastal Mediterranean',
                                                              'notes':'created in profiles.ipynb'})

RECCAP['custom_masks'].to_netcdf('../../supporting_data/RECCAP2_region_masks_customized.nc')