import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helpers import *

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams.update({'font.family':'Arial'})

from acgc import figstyle
figstyle.grid_on()

design_dict = {'Americas': {'color': '0.2', 'zorder': 1, 'label': 'Americas', 'alpha': 1},
               'Africa and Middle East': {'color': '0.4', 'zorder': 2, 'label': 'Africa + Middle East', 'alpha': 1},
               'Asia and Oceania': {'color': '0.6', 'zorder': 3, 'label': 'Asia + Oceania', 'alpha': 1},
               'Europe and Former USSR': {'color': '0.8', 'zorder': 4, 'label': 'Europe + Former USSR', 'alpha': 1},
               'Other Metals Production':{'color':'0.3', 'zorder':1, 'label':'Other Metals Production', 'alpha':1}, 
               'Gold and Silver Production':{'color':'0.5', 'zorder':2, 'label':'Gold and Silver Production', 'alpha':1}, 
               'Fossil-Fuel Combustion':{'color':'0.7', 'zorder':3, 'label':'Fossil Fuel Combustion', 'alpha':1}, 
               'Mercury Production and Use':{'color':'0.9', 'zorder':4, 'label':'Mercury Production and Use', 'alpha':1},
               }

# update colors
for color, key in zip(["#0072BD", "#D95319", [0.7,0.7,0.7], "#EDB120"], 
                      ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR']):
    design_dict[key]['color'] = color

# function to make emission plot
def make_emission_plot3(df, ax, groups=[], 
    colors=['tab:blue', 'tab:orange', 'darkgray', 'gold'], labels=[],
    xmin=1500, xmax=2300, ymin=0, ymax=14000, fontsize=10, legend=True):
    
    # make stackplot of emissions
    ax.stackplot(df['Year'], df[groups].T, labels=labels,
                 colors=colors, 
                 zorder=4)
    ax.set_xlabel('Year', fontsize=fontsize)
    ax.set_ylabel('Hg Emissions and Releases (Gg a$^{-1}$)', fontsize=fontsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # square legend edges
    if legend==True:
        ax.legend(edgecolor='None', facecolor='whitesmoke', 
                framealpha=1, fontsize=fontsize, fancybox=False, 
                loc='upper center',
                bbox_to_anchor=(0.5, 1.4), ncol=2)

    ax.grid(ls='-', color='whitesmoke', zorder=1)

    ax.axvline(2010, 0, 1, lw=0.5, color='0.3', linestyle='--', zorder=0)

    return

def add_annotated_text(ax, x, y, text, fontsize=10, ha='center', va='center', color='0.2', zorder=6):
    ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, color=color, zorder=zorder)
    return

def add_annotated_arrow(ax, x1, y1, x2, y2, arrowstyle='<-, head_width=0.15', lw=0.5, color='0.2', zorder=6):
    ax.annotate('', xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(color=color, arrowstyle=arrowstyle, lw=lw), zorder=zorder)
    return

def modify_ticks(ax):
    c = '0.2'
    # set tick spines on top and bottom
    ax.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False)
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
    ax.tick_params(axis='x', which='both', colors=c)
    ax.tick_params(axis='y', which='both', colors=c)
    return

# --------------------------------------------------------------------------------
# Make figure 2
# --------------------------------------------------------------------------------
palette = [
        [63, 63, 69], [42, 42, 52], [31, 32, 36], [123, 131, 125], [173, 168, 153],
        [83, 83, 79], [159, 179, 185], [195, 212, 217], [240, 249, 249], [144, 90, 77], [62, 46, 42],
        ]
palette_rgb = [np.array(p)/255 for p in palette]

fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

# update colors
for color, key in zip([palette_rgb[3], palette_rgb[6], palette_rgb[1], palette_rgb[9],],
                      ['Other Metals Production', 'Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production and Use']):
    design_dict[key]['color'] = color

sectors = ['Other Metals Production', 'Gold and Silver Production', 'Fossil-Fuel Combustion', 'Mercury Production and Use']

for color, key in zip(["#0072BD", "#D95319", [0.7,0.7,0.7], "#EDB120"], 
                      ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR']):
    design_dict[key]['color'] = color

regions = ['Americas', 'Africa and Middle East', 'Asia and Oceania', 'Europe and Former USSR']

# -- plot sector emissions
for scenario, ax, bool_legend in zip(['SSP1-26', 'SSP5-85'], [ax1, ax2], [False, False]):
    df = load_800_year(category='sector', scenario=scenario)
    colors = [design_dict[group]['color'] for group in sectors]
    # -----------------------------------------------------------------------
    xmin, xmax = 1500, 2300
    ymin, ymax = 0, 15000
    make_emission_plot3(df, ax, groups=sectors, colors=colors, labels=[design_dict[group]['label'] for group in sectors],
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fontsize=8, legend=bool_legend)
    
    # -- add annotations
    f_y = 0.94 # y position for text
    f_y_arrow = 0.88 # y position for arrow

    # - arrow and label for pre-2010
    x_text = 2000
    x_arrow1 = x_text+5
    x_arrow2 = 1730
    arrowstyle = '<-, head_width=0.15'
    pre_1510_total = 68 # add 68 Gg for pre-1500 emissions
    pre_2010_total = int(np.round(get_cumulative_emissions(df[df['Year']<=2010], sectors).sum(),0)+pre_1510_total)
    add_annotated_text(ax, x_text, f_y*ymax, f"Pre-2010: {pre_2010_total} Gg", fontsize=8, ha='right')
    add_annotated_arrow(ax, x_arrow1, f_y_arrow*ymax, x_arrow2, f_y_arrow*ymax, arrowstyle=arrowstyle)

    # - arrow and label for post-2010
    x_text = 2020
    x_arrow1 = x_text-5
    post_2010_total = int(np.round(get_cumulative_emissions(df[df['Year']>=2010], sectors, include_first_year=False).sum(),0))
    # check if post-2010 total is greater than 1000 Gg -- prevents arrow from hanging past text when total is small
    if post_2010_total >= 1000:
        x_arrow2 = x_arrow1+256
    else:
        x_arrow2 = x_arrow1+246
    add_annotated_text(ax, x_text, f_y*ymax, f"Post-2010: {post_2010_total} Gg", fontsize=8, ha='left')
    add_annotated_arrow(ax, x_arrow1, f_y_arrow*ymax, x_arrow2, f_y_arrow*ymax)

# -- plot regional emissions
for scenario, ax, bool_legend in zip(['SSP1-26', 'SSP5-85'], [ax3, ax4], [False, False]):
    df = load_800_year(category='region', scenario=scenario)
    colors = [design_dict[group]['color'] for group in regions]
    # -----------------------------------------------------------------------
    xmin, xmax = 1500, 2300
    ymin, ymax = 0, 15000
    make_emission_plot3(df, ax, groups=regions, colors=colors, labels=[design_dict[group]['label'] for group in regions],
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fontsize=8, legend=bool_legend)
    
    # - arrow and label for pre-2010
    x_text = 2000
    x_arrow1 = x_text+5
    x_arrow2 = 1730
    arrowstyle = '<-, head_width=0.15'
    pre_1510_total = 68 # add 68 Gg for pre-1500 emissions
    pre_2010_total = int(np.round(get_cumulative_emissions(df[df['Year']<=2010], regions).sum(),0)+pre_1510_total)
    add_annotated_text(ax, x_text, f_y*ymax, f"Pre-2010: {pre_2010_total} Gg", fontsize=8, ha='right')
    add_annotated_arrow(ax, x_arrow1, f_y_arrow*ymax, x_arrow2, f_y_arrow*ymax, arrowstyle=arrowstyle)

    # - arrow and label for post-2010
    x_text = 2020
    x_arrow1 = x_text-5
    post_2010_total = int(np.round(get_cumulative_emissions(df[df['Year']>=2010], regions, include_first_year=False).sum(),0))
    # check if post-2010 total is greater than 1000 Gg -- prevents arrow from hanging past text when total is small
    if post_2010_total >= 1000:
        x_arrow2 = x_arrow1+256
    else:
        x_arrow2 = x_arrow1+246
    add_annotated_text(ax, x_text, f_y*ymax, f"Post-2010: {post_2010_total} Gg", fontsize=8, ha='left')
    add_annotated_arrow(ax, x_arrow1, f_y_arrow*ymax, x_arrow2, f_y_arrow*ymax)

yticks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]

for ax in [ax1, ax2, ax3, ax4]:
    modify_ticks(ax)
    ax.set_yticks(yticks, labels=[f'{int(y/1000)}' for y in yticks], minor=False)

# set minor ticks
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_yticks(np.arange(0, 16000, 1000), minor=True)
    ax.set_xticks(np.arange(1500, 2301, 100), minor=False)
    ax.set_xticks(np.arange(1500, 2300, 50), minor=True)

for ax in [ax1, ax3]:
    ax.set_ylabel('Hg Emissions and Releases (Gg a$^{-1}$)', c='0.2')
for ax in [ax2, ax4]:
    ax.set_ylabel('')

for ax in [ax3, ax4]:
    ax.set_xlabel('Year', c='0.2')
for ax in [ax1, ax2]:
    ax.set_xlabel('')

for ax, letter, y in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd'], [0.98, 0.97, 0.98, 0.97]):
    ax.text(s=letter, x=0.02, y=y, transform=ax.transAxes, fontsize=14, va='top', ha='left', weight='semibold')

for ax, txt in zip([ax1, ax2, ax3, ax4], ['SSP1-2.6', 'SSP5-8.5', 'SSP1-2.6', 'SSP5-8.5']):
    ax.text(s=txt, x=0.03, y=0.2, transform=ax.transAxes, fontsize=14, va='top', ha='left', c='0.3')
    ax.grid(lw=0.5, color='0.5', which='major')
    ax.grid(lw=0.3, color='0.5', which='minor')

# put ax2.yticklabels on the right without removing yticks on the left
ax2.tick_params(axis='y', left=True, right=True, which='both', labelleft=False, labelright=True)
# put ax4.yticklabels on the right without removing yticks on the left
ax4.tick_params(axis='y', left=True, right=True, which='both', labelleft=False, labelright=True)

handles_sector, labels_sector = ax2.get_legend_handles_labels()
handles_region, labels_region = ax4.get_legend_handles_labels()    
handles = []
labels = []
for i, j in zip(handles_sector, handles_region):
    handles.append(i)
    handles.append(j)
for i, j in zip(labels_sector, labels_region):
    labels.append(i)
    labels.append(j)

sector_legend = fig.legend(handles_sector, labels_sector, loc='center left', bbox_to_anchor=(0.07, 0.85), ncol=1, fontsize=7.5, handlelength=1, edgecolor='None', facecolor='whitesmoke', framealpha=1)
region_legend = fig.legend(handles_region, labels_region, loc='center left', bbox_to_anchor=(0.07, 0.35), ncol=1, fontsize=7.5, handlelength=1, edgecolor='None', facecolor='whitesmoke', framealpha=1)

plt.savefig(f'../figures/figure_1.png', dpi=1200, bbox_inches='tight')
plt.savefig(f'../figures/figure_1.pdf',format='pdf', bbox_inches='tight')