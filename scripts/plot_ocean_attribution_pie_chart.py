import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from acgc import figstyle
from helpers import subset_single
from helpers_colormaps import get_nature_cmaps
import json

km3_to_L   = 1e12    # km3 -> L
kg_to_mol  = 1e3/200.59 # kg Hg -> mol Hg
surfintvol = 4.9e20 # L (surface + intermediate ocean volume; 0-1500m depth; GEBCO bathymetry)
deepvol    = 8.3e20 # L (deep ocean volume; >1500m depth; GEBCO bathymetry)

year_sel = 2010

df = pd.read_csv('../output/main/time_attribution/2010/all_inputs_output_sector_SSP1-26_1509_1510.csv')
natural = subset_single(df, match_column='Year', match_values=[-2000])
pre_1510 = subset_single(df, match_column='Year', match_values=[year_sel])
pre_1510 -= natural.values

# -- set up output dictionary
# make each column of df a key with an empty list as value
output_dict = {col: [] for col in df.columns[1:]}
output_dict['period'] = []

# -- add natural steady state values
output_dict['period'].append('natural')
for col in df.columns[1:]:
    output_dict[col].append(natural[col].values[0])

# -- add pre-1510 values
output_dict['period'].append('pre-1510')
for col in df.columns[1:]:
    output_dict[col].append(pre_1510[col].values[0])

# -- loop over periods and add values to output_dict
periods = ['1510_1600', '1600_1700', '1700_1800', '1800_1900', '1900_2000', '2000_2010']
for period in periods:
    df = pd.read_csv(f'../output/main/time_attribution/2010/all_inputs_output_sector_SSP1-26_{period}.csv')
    subset = subset_single(df, match_column='Year', match_values=[year_sel])
    subset -= natural.values
    output_dict['period'].append(period)
    for col in df.columns[1:]:
        output_dict[col].append(subset[col].values[0])
    del df

output = pd.DataFrame(output_dict)
output['upper_ocean'] = output[['ocs','oci']].sum(axis=1)
output['deep_ocean'] = output[['ocd']].sum(axis=1)

# ---------------   
# pre-1800 anthropogenic contributions to ocean are small, so we'll group them
# ---------------

# group so that ['pre-1510','1510_1600','1600_1700','1700_1800'] are in the same row
pre_1800 = output.loc[
    output['period'].isin(['pre-1510', '1510_1600', '1600_1700', '1700_1800']), 
    output.columns.difference(['period'])
].sum().to_frame().T
pre_1800['period'] = 'pre-1800'

output = pd.concat([output, pre_1800], ignore_index=True)
output.drop(index=output[output['period'].isin(['pre-1510','1510_1600','1600_1700','1700_1800'])].index, inplace=True)
# assign following period order: ['natural','pre-1800','1800_1900','1900_2000','2000_2010']
order = ['natural','pre-1800','1800_1900','1900_2000','2000_2010']
output['period'] = pd.Categorical(output['period'], categories=order, ordered=True)
output = output.dropna(subset=['period']).sort_values('period').reset_index(drop=True)
output = output.reset_index(drop=True)

output['upper_ocean_pM'] = output['upper_ocean'] * (1e3*kg_to_mol*1e12/surfintvol)
output['deep_ocean_pM'] = output['deep_ocean'] * (1e3*kg_to_mol*1e12/deepvol)

def plot_pie(ax, data, labels, draw_labels=True, draw_percentages=True, legend=True):
    # -- 
    # label the fracation of total within each wedge
    fractions = data / np.sum(data)
    # convert fractions to percentage
    fractions = ["{:.0%}".format(f) for f in fractions]

    # put fraction in 
    if draw_labels == False:
        text_labels = None
    else:
        text_labels = labels
    
    color_list = [(20, 17, 65), (58, 80, 138), (109, 130, 148), (161, 171, 141), (243, 243, 216)]
    color_list = [(r/255, g/255, b/255) for r, g, b in color_list]

    wedges, texts, autotexts = ax.pie(data, wedgeprops=dict(width=1, edgecolor='k', lw=0.3), 
                           autopct='%1.0f%%',
                           startangle=90,
                           colors = color_list[::-1],
                           labels=text_labels,
    )

    if draw_percentages:
        # specify colors of autotexts
        for autotext in autotexts[2:]:
            autotext.set_color('w')
        for autotext in autotexts[0:2]:
            autotext.set_color('k')
        # replace autotext with '' if value is less than 5
        for autotext in autotexts:
            if float(autotext.get_text().replace('%','')) < 5:
                autotext.set_text('')
        # draw percentages
        plt.setp(autotexts, size=10)
    
    # horizontal legend
    if legend:
        ax.legend(wedges[::-1], labels[::-1],
                  loc='center right', bbox_to_anchor=(1., 0, 0.5, 1), 
                  ncol=1, fontsize=10, edgecolor='0.3')

    return wedges, texts, autotexts

label_dict = {
    'natural':  'Natural',
    'pre-1800': 'Pre-1800',
    '1800_1900': '1800 - 1900',
    '1900_2000': '1900 - 2000',
    '2000_2010': '2000 - 2010'
}

labels = [label_dict[p] for p in output['period'].dropna().values]

draw_labels = False
fig = plt.figure(figsize=(7, 3.5))
# add gridspec
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

ax0 = fig.add_subplot(gs[0])
wedges, texts, autotexts = plot_pie(ax0, data=output['atm'].values[::-1], labels=labels[::-1], draw_labels=draw_labels, legend=False)
ax0.text(0.05, 0.88, 'a', transform=ax0.transAxes, fontsize=16, va='top', ha='left', fontweight='bold',)
ax0.text(0.5, 0.95, 'Atmosphere', transform=ax0.transAxes, fontsize=11, va='bottom', ha='center')

ax1 = fig.add_subplot(gs[1])
wedges, texts, autotexts = plot_pie(ax1, data=output['upper_ocean'].values[::-1], labels=labels[::-1], draw_labels=draw_labels, legend=False)
# add letter a to upper left corner
ax1.text(0.05, 0.88, 'b', transform=ax1.transAxes, fontsize=16, va='top', ha='left', fontweight='bold', )
ax1.text(0.5, 0.95, 'Upper Ocean (0 - 1500 m)', transform=ax1.transAxes, fontsize=11, va='bottom', ha='center')

ax2 = fig.add_subplot(gs[2])
wedges, texts, autotexts = plot_pie(ax2, data=output['deep_ocean'].values[::-1], labels=labels[::-1], draw_labels=draw_labels, legend=False)
# add letter b to upper left corner
ax2.text(0.05, 0.88, 'c', transform=ax2.transAxes, fontsize=16,  va='top', ha='left', fontweight='bold',)
# add title and place it immediately above the plot
ax2.text(0.5, 0.95, 'Deep Ocean (\u003E1500 m)', transform=ax2.transAxes, fontsize=11, va='bottom', ha='center')

# add additional axis to figure (bottom) and put common legend there
ax3 = fig.add_subplot(gs[0, :])
ax3.axis('off')

# add legend
ax3.legend(wedges[::-1], labels, loc='center', bbox_to_anchor=(0.25, 0.15, 0.5, 0), ncol=5, fontsize=10, edgecolor='0.7', fancybox=False)

plt.savefig('../figures/figure_S8.pdf', bbox_inches='tight')
plt.savefig('../figures/figure_S8.png', dpi=1200, bbox_inches='tight')