import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from acgc import figstyle
import json
with open('../profiles/supporting_data/nature_reviews.json', 'r') as f:
    nature_cmaps = json.load(f)

color_dict = {c: nature_cmaps[c] for c in nature_cmaps.keys()}

# convert from rgb [0 - 255] to [0 - 1]
for k, v in color_dict.items():
    color_dict[k] = np.array(v) / 255

add_reccap_masks = False

style_dict = { 'Arctic':{'marker':'o', 'color':color_dict['purple'][4], 'edgecolor':'white'},
              'Coastal Arctic': {'marker':'D', 'color':color_dict['purple'][2], 'edgecolor':'white'},
              'Land':{'marker':'s', 'color': '0.9', 'edgecolor':'black'}, 
              'Atlantic':{'marker':'o', 'color':color_dict['blue'][4], 'edgecolor':'white'},
              'Coastal Atlantic': {'marker':'D', 'color':color_dict['blue'][2], 'edgecolor':'white'},
              'Indian':{'marker':'o', 'color':color_dict['yellow'][4], 'edgecolor':'white'},
              'Coastal Indian': {'marker':'D', 'color':color_dict['yellow'][2], 'edgecolor':'white'},
              'Mediterranean':{'marker':'o', 'color':color_dict['stone'][4], 'edgecolor':'white'},
              'Coastal Mediterranean': {'marker':'D', 'color':color_dict['stone'][2], 'edgecolor':'white'},
              'Pacific':{'marker':'o', 'color':color_dict['red'][4], 'edgecolor':'white'},
              'Coastal Pacific': {'marker':'D', 'color':color_dict['red'][2], 'edgecolor':'white'},
              'Southern':{'marker':'o', 'color':color_dict['green'][4], 'edgecolor':'white'},
              'Coastal Southern': {'marker':'D', 'color':color_dict['green'][2], 'edgecolor':'white'},
}

df = pd.read_csv('../profiles/output/seawater_HgT_observation_compilation.csv')
df = df[df['quantity'].isin(['Hg_T','Hg_T_D'])]

# create map of profiles
fig = plt.figure(figsize=(7.5, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='0.2', lw=0.5, zorder=2)
# add gridlines - but not right or bottom ones
gl = ax.gridlines(draw_labels=True, linestyle='--', lw=0.5, alpha=0.1, zorder=1)
gl.xlabel_style = {'size': 8, 'color': '0.1'}
gl.ylabel_style = {'size': 8, 'color': '0.1'}

# plot bathymetry
bath = xr.open_mfdataset('../profiles/supporting_data/GEBCO_regridded_all.nc')
im = bath['elevation'].plot(cmap='Greys_r', add_colorbar=False, vmax=0, vmin=-7000, alpha=0.6)

# create colorbar for im and set ticks
cbar = plt.colorbar(im, orientation='vertical', shrink=0.6, pad=0.03, aspect=30)
cbar.set_label('Depth (km)')
cbar.set_ticks(np.arange(-7000, 1, 1000))
cbar.set_ticklabels(np.arange(7, -1, -1))

# plot data
for ID in df['uniqueID'].unique():
    sel = df[df['uniqueID'] == ID]
    if sel.quantity.values[0] in ['Hg_T', 'Hg_T_D']:
        plt.plot(sel.lon.values[0], sel.lat.values[0], 
                 markersize=5,
                 markerfacecolor=style_dict[sel.basin.values[0]]['color'],
                 marker=style_dict[sel.basin.values[0]]['marker'], 
                 markeredgecolor=style_dict[sel.basin.values[0]]['edgecolor'], 
                 markeredgewidth=0.2,
                 transform=ccrs.PlateCarree())

if add_reccap_masks == True:
    basin_label_dict = {0: 'Land', 1: 'Atlantic', 2: 'Pacific', 3: 'Indian', 4: 'Arctic', 5: 'Southern', 6: 'Mediterranean',
                        11: 'Coastal Atlantic', 12: 'Coastal Pacific', 13: 'Coastal Indian', 14: 'Coastal Arctic', 15: 'Coastal Southern', 16: 'Coastal Mediterranean'}
    RECCAP2 = xr.open_mfdataset('../profiles/supporting_data/RECCAP2_region_masks_customized.nc')
    for i in basin_label_dict.keys():
        basin = basin_label_dict[i]
        cmap = LinearSegmentedColormap.from_list('tmp_cmap', ['k',style_dict[basin]['color'], 'k'])
        RECCAP2['custom_masks'].where(RECCAP2['custom_masks']==i).plot(cmap=cmap, alpha=0.3, add_colorbar=False, zorder=1)

labels  = style_dict.keys()
handles = [plt.Line2D([0], [0], marker=style_dict[basin]['marker'], linestyle='None', markerfacecolor=style_dict[basin]['color'], markeredgecolor=style_dict[basin]['edgecolor'], markeredgewidth=0.2, markersize=5, label=basin) for basin in labels]
for h in handles:
    if 'Land' in h.get_label():
        h.set_label('Other')
plt.legend(handles=handles, ncols=6, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, -0.25), edgecolor='0.8', facecolor='0.98') #loc='center right', )#bbox_to_anchor=(1.2, 0.5))

if add_reccap_masks == True:
    plt.savefig('../figures/figure_S1.png', dpi=1200, bbox_inches='tight')
else:
    plt.savefig('../figures/figure_S1.png', dpi=1200, bbox_inches='tight')
