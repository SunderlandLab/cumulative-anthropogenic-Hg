import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from acgc import figstyle

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
    return

# -------------------------------------------------------------------------------------------------------------
# -- load and prepare data
# -------------------------------------------------------------------------------------------------------------
df = pd.read_csv('../profiles/output/interpolated_seawater_HgT_observation_compilation.csv')
volumes = pd.read_csv('../profiles/output/basin_volumes_by_depth.csv')

quantity_keys=['Hg_T', 'Hg_T_D', 'Hg_T_D_fish']
groupby_keys=['basin', 'depth']

# subset the dataframe to only include the quantity_keys
df = df[df['quantity'].isin(quantity_keys)]

# define dictionary of summary statistics to calculate
lambda_dict = {'min' : lambda x: np.min(x),
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

# -- define order of basins to plot
group_order = ['Arctic', 'Atlantic', 'Indian', 'Pacific', 'Southern',  'Land', 
              'Coastal Arctic', 'Coastal Atlantic', 'Coastal Indian', 'Coastal Pacific', 'Coastal Southern', 'Coastal Mediterranean',]
n_groups = len(group_order)


# ------------------- plot depth profiles of total Hg -------------------
# create figure
fontsize = 7
n_col = 6
n_row = 2

fig = plt.figure(figsize=(7.5, 5))
gs = fig.add_gridspec(n_row, n_col)
ax_list = []

# loop over each basin and plot depth profile
for i, name in enumerate(group_order):
    col = i % n_col
    row = i // n_col
    ax = plt.subplot(gs[row, col])
    ax_list.append(ax)
    
    subset = conc[conc['basin']==name]

    # ---------- plot volume profile ----------
    # plot volume profile
    ax2 = ax.twiny()
    # get total volume of water in basin
    total_volume = subset['volume [L]'].sum()
    # plot fraction of total volume
    ax2.plot(subset['volume [L]']/total_volume, subset['depth'], c='steelblue', lw=1.5, zorder=1)
    ax2.set_xticks([])

    # ---------- plot concentration profile ----------
    # plot depth profile of total Hg
    ax.plot(subset['p50'], subset['depth'], c='0.2', lw=1.5, zorder=10)
    # fill between 25th and 75th percentiles
    ax.fill_betweenx(subset['depth'], subset['p25'], subset['p75'], color='0.7', lw=0, alpha=0.6, zorder=5)
    # fill between 5th and 95th percentiles
    ax.fill_betweenx(subset['depth'], subset['p5'], subset['p95'], color='0.7',lw=0, alpha=0.2, zorder=3)

    # add label in upper right corner
    if name != 'Land':
        label = name
    else:
        label = 'Other'

    # put title in bottom right corner and add line break to labels with spaces
    label = label.replace(' ', '\n')
    # add total volume to label in scientific notation (rounded to 2 decimal places)
    label += f'\n{total_volume:.1e} L'
    
    ax.text(0.96, 0.03, label, ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize, c='0.2')
    
    if i in [0, 6]:
        pass
    elif i in [5, 11]:
        # put y-axis label on right
        ax.yaxis.tick_right()
    else:
        ax.set_yticklabels([])

    ax.set_xticks([0, 1.5, 3])
    ax.set_xticklabels([0, 1.5, 3])
    ax.set_xlim(0, 3)

    if name == 'Land':
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 2.5, 5])
        ax.set_xticklabels([0, 2.5, 5])

    # get current x-lim of ax2
    xlim = ax2.get_xlim()
    # set x-min to 0
    ax2.set_xlim(0, xlim[1])
    
    # reverse y-axis
    ax.set_ylim(6000, 0)
    ax2.set_ylim(6500, 0)

    ax.grid(False)
    ax2.grid(False)
    # modify ticks
    modify_ticks(ax)
    
    # add letter to bottom left
    ax.text(0.05, 0.03, f'{chr(97+i)}', ha='left', va='bottom', transform=ax.transAxes, fontsize=fontsize+3, fontweight='bold')
    
fig.supxlabel('Total Hg [pmol L$^{-1}$]', fontsize=fontsize+3, c='0.2')
# add supylabel to left and right
fig.supylabel('Depth [m]', fontsize=fontsize+3, c='0.2')

# save figure
fig.savefig('../figures/figure_S2.pdf', bbox_inches='tight', format='pdf')
fig.savefig('../figures/figure_S2.png', bbox_inches='tight', dpi=1200)

print_caption = False
if print_caption:
    # print display caption for this figure
    print('Figure S2: Depth profiles of total mercury concentration (pmol L-1) by ocean basin.')
    print('The black line represents the median concentration, and grey shading represents')
    print('the interquartile range (dark grey) and 5th to 95th percentile range (light grey).')
    print('The relative volume profile in each basin is shown in blue, and the total volume of ')
    print('the basin is listed in the bottom right corner of each plot.')
