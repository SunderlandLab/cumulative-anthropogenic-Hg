import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from acgc import figstyle

c = '0.1'

# --- look at emission fluxes attributable to L/W releases
from make_evasion_comparison_tables import make_evasion_budget

# -- 1. get natural flux (legacy + primary)
df_natural = pd.DataFrame(columns=['Year', 'Scenario', 'Natural Flux'])
for scenario in ['SSP1-26', 'SSP5-85']:
    for yr in range(1850, 2310, 10):
        tmp = make_evasion_budget(path='../output/main/attribution/', reservoir_fn_prefix='output', scenario=scenario, sel_year=yr).groupby(by=['Emission Type', 'Year'], as_index=False).sum(numeric_only=True)
        tmp = tmp[tmp['Emission Type'].str.contains('natural')]
        tmp['Scenario'] = scenario
        tmp = tmp.groupby(by=['Year', 'Scenario'], as_index=False).sum(numeric_only=True)
        tmp.rename(columns={'Flux': 'Natural'}, inplace=True)
        tmp = tmp[['Year', 'Scenario', 'Natural']]
        df_natural = pd.concat([df_natural, tmp])

# -- 2. get anthropogenic LW legacy emission flux
df_LW = pd.DataFrame(columns=['Year', 'Scenario', 'Legacy (LW)'])
for scenario in ['SSP1-26', 'SSP5-85']:
    for yr in range(1850, 2310, 10):
        tmp = make_evasion_budget(path='../output/main/attribution/', reservoir_fn_prefix='output', scenario=scenario, sel_year=yr, match_dict={'media':['LW']}).groupby(by=['Emission Type', 'Year'], as_index=False).sum(numeric_only=True)
        legacy_anthro = tmp[tmp['Emission Type'] == 'legacy anthropogenic']['Flux'].item()
        tmp = pd.DataFrame({'Year': [yr], 'Scenario': [scenario], 'Legacy (LW)': [legacy_anthro]})
        df_LW = pd.concat([df_LW, tmp])

# -- 3. get anthropogenic air emission flux
df_air = pd.DataFrame(columns=['Year', 'Scenario', 'Legacy (Air)', 'Primary (Air)'])
for scenario in ['SSP1-26', 'SSP5-85']:
    for yr in range(1850, 2310, 10):
        tmp = make_evasion_budget(path='../output/main/attribution/', reservoir_fn_prefix='output', scenario=scenario, sel_year=yr, match_dict={'media':['Air']}).groupby(by=['Emission Type', 'Year'], as_index=False).sum(numeric_only=True)
        legacy_anthro = tmp[tmp['Emission Type'] == 'legacy anthropogenic']['Flux'].item()
        primary_anthro = tmp[tmp['Emission Type'] == 'primary anthropogenic']['Flux'].item()
        tmp = pd.DataFrame({'Year': [yr], 'Scenario': [scenario], 'Legacy (Air)': [legacy_anthro], 'Primary (Air)': [primary_anthro]})
        df_air = pd.concat([df_air, tmp])

# merge dataframes
df = pd.merge(df_natural, df_LW, on=['Year', 'Scenario'])
df = pd.merge(df, df_air, on=['Year', 'Scenario'])

def modify_ticks(ax):
    c = '0.1'
    # set spine and tick color
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
    # set tick color 
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    return

year_selection = [1850, 1970, 2010, 2100, 2200, 2300]

category_order = ['Natural', 'Legacy (LW)', 'Legacy (Air)', 'Primary (Air)']
colors = colors = [(100, 100, 100), (155, 155, 114), (207, 220, 236), (245, 249, 254)] #[(250, 250, 250), (155, 155, 114), (207, 220, 236), (245, 249, 254)]
colors = [(r/255, g/255, b/255) for r, g, b in colors]

fig = plt.figure(figsize=(6.5, 2.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0])
# plot bar chart of emissions by type
data = df[(df['Year'].isin(year_selection)) & (df['Scenario'] == 'SSP1-26')]

# plot stacked bar chart
bottom = np.repeat(0., len(year_selection))
for j, cat in enumerate(category_order):
    ax1.bar(np.arange(len(year_selection)), height=data[cat].values, color=colors[j], label=cat, bottom=bottom, zorder=2)
    bottom += data[cat].values
# put border around bars
ax1.bar(np.arange(len(year_selection)), height=bottom, color='none', edgecolor='black', linewidth=0.5, zorder=3)

import matplotlib.ticker as ticker
ax1.set_xticks(range(len(year_selection)))
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.set_xticklabels(year_selection, c=c)
ax1.set_ylabel('Emission Flux (Gg a$^{-1}$)', c=c)
# convert y-axis to Gg a-1 from Mg a-1
yticks = ax1.get_yticks()
ax1.set_yticks(yticks)  # set the tick positions explicitly
ax1.set_yticklabels([f'{y/1000:.0f}' for y in yticks], c=c)
ax1.grid(False)

# add annotations
ax1.text(0.95, 0.95, 'SSP1-2.6', transform=ax1.transAxes, ha='right', va='top', fontsize=14)
ax1.text(0.03, 0.95, 'a', transform=ax1.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')
modify_ticks(ax1)

ax2 = fig.add_subplot(gs[1])
# plot bar chart of emissions by type
data = df[(df['Year'].isin(year_selection)) & (df['Scenario'] == 'SSP5-85')]
# plot stacked bar chart
bottom = np.repeat(0., len(year_selection))
for j, cat in enumerate(category_order):
    ax2.bar(np.arange(len(year_selection)), height=data[cat].values, color=colors[j], label=cat, bottom=bottom, zorder=2)
    bottom += data[cat].values
# put border around bars
ax2.bar(np.arange(len(year_selection)), height=bottom, color='none', edgecolor='black', linewidth=0.5, zorder=3)

ax2.set_xticks(range(len(year_selection)), minor=False)
ax2.xaxis.set_minor_locator(ticker.NullLocator())
ax2.set_xticklabels(year_selection, c=c)
ax2.set_ylabel('')
# convert y-axis to Gg a-1 from Mg a-1
yticks = ax2.get_yticks()
ax2.set_yticks(yticks)  # set the tick positions explicitly
ax2.set_yticklabels([f'{y/1000:.0f}' for y in yticks], c=c)
ax2.grid(False)

# add annotations
ax2.text(0.95, 0.95, 'SSP5-8.5', transform=ax2.transAxes, ha='right', va='top', fontsize=14)
ax2.text(0.03, 0.95, 'b', transform=ax2.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')

# update legend elements to have black frame
handles, labels = ax1.get_legend_handles_labels()
# Create new lists for legend handles and labels
legend_handles = []
for handle in handles:
    # Check if the handle contains patches (like bar plots)
    if hasattr(handle, 'patches'):  # this works for BarContainers
        patches = []
        for patch in handle.patches:  # access individual patches
            # Copy the patch and modify edgecolor/linewidth
            new_patch = mpatches.Patch(facecolor=patch.get_facecolor(), edgecolor='black', linewidth=0.5)
            patches.append(new_patch)
        # Append the new patch list to legend handles
        legend_handles.append(patches[0])

# make common legend below plots
fig.legend(handles=legend_handles, labels=labels, ncol=4, fontsize=10, edgecolor='0.9', labelcolor='0.1', frameon=True, fancybox=False, loc='lower center', bbox_to_anchor=(0.5, -0.15))
plt.savefig('../figures/figure_4.pdf', bbox_inches='tight')
plt.savefig('../figures/figure_4.png', bbox_inches='tight', dpi=1200)
