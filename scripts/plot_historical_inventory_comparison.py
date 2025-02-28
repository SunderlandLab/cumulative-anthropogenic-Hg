import numpy as np 
import pandas as pd 
from acgc import figstyle
import matplotlib.pyplot as plt
from helpers import modify_ticks, get_cumulative_emissions

streets = pd.read_csv('../inputs/misc/literature_inventories/Streets_2019_500_year_with_bounds.csv', header=2)

# read Hylander and Meili 2019
hylander = pd.read_excel('../inputs/misc/literature_inventories/Hylander_and_Meili_2002.xlsx', header=2)
hylander = hylander.iloc[:-2] # remove final two rows
# subset global total and year
hylander = hylander[['Year','Globally mined']]
hylander['Year'] = hylander['Year'].astype(str)
# where Year is a range, split it into two columns. Use conditional to check if it is a range
hylander['start_year'] = hylander['Year'].apply(lambda x: x.split('-')[0] if '-' in x else x)
hylander['end_year'] = hylander['Year'].apply(lambda x: x.split('-')[1] if '-' in x else x)
# calculate number of years in range
hylander['n_years'] = hylander['end_year'].astype(int) - hylander['start_year'].astype(int) + 1
# calculate quantity mined per year
hylander['Globally mined [Mg/yr]'] = hylander['Globally mined'] / hylander['n_years']
# expand the dataframe to have one row per year
hylander = hylander.loc[hylander.index.repeat(hylander['n_years'])]
hylander['Year'] = hylander['start_year'].astype(int) + hylander.groupby(level=0).cumcount()
hylander = hylander.drop(columns=['start_year','end_year','n_years'])

# read Guerrero and Schneider (2023)
guerrero = pd.read_excel('../inputs/misc/literature_inventories/Guerrero_and_Schneider_2023_Table_S1.xlsx', header=2)
guerrero.rename(columns={'Unnamed: 0':'Period', 'Unnamed: 15': 'Hg SP'}, inplace=True)
guerrero = guerrero.iloc[:-2] # remove final two rows
guerrero = guerrero[['Period','Hg SP']] # subset Hg SP and Period
# where Period is a range, split it into two columns. Use conditional to check if it is a range
guerrero['start_year'] = guerrero['Period'].apply(lambda x: x.split('-')[0] if '-' in x else x)
guerrero['end_year'] = guerrero['Period'].apply(lambda x: x.split('-')[1] if '-' in x else x)
# calculate number of years in range
guerrero['n_years'] = guerrero['end_year'].astype(int) - guerrero['start_year'].astype(int) + 1
# calculate quantity mined per year
guerrero['Hg SP [Gg/yr]'] = guerrero['Hg SP'] / guerrero['n_years']
# expand the dataframe to have one row per year
guerrero = guerrero.loc[guerrero.index.repeat(guerrero['n_years'])]
guerrero['Year'] = guerrero['start_year'].astype(int) + guerrero.groupby(level=0).cumcount()
guerrero = guerrero.drop(columns=['start_year','end_year','n_years'])
guerrero['Hg SP [Mg/yr]'] = guerrero['Hg SP [Gg/yr]'] * 1e3

guerrero_2 = pd.read_excel('../inputs/misc/literature_inventories/Guerrero_and_Schneider_2023_Table_S11.xlsx', header=1)
guerrero_2 = guerrero_2.iloc[:-2]
guerrero_2['Hg SP [Mg/yr]'] = guerrero_2['Global Hg SP*'] * 1e3
guerrero_2['Year'] = guerrero_2['Year'].astype(int)

# concatenate the two Guerrero dataframes
guerrero = pd.concat([guerrero[['Year','Hg SP [Mg/yr]']], guerrero_2[['Year','Hg SP [Mg/yr]']]], axis=0)

# --- make the figure
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(111)
# -- plot Streets et al. 2019
streets_sector = pd.read_csv('../inputs/emissions/SSP5-85_sector_1510_2300.csv')
# merge in streets[['Year','scale_lower','scale_upper']] on Year
streets_sector = pd.merge(streets_sector, streets[['Year','scale_lower','scale_upper']], on='Year', how='left')

col_list = [
'Air - Gold and Silver Production', 'Air - Mercury Production and Use',
'LW - Gold and Silver Production', 'LW - Mercury Production and Use',]

streets_sel = streets_sector.copy()
streets_sel = streets_sel[(streets_sel['Year']>=1500) & (streets_sel['Year']<=2010)]
streets_sel['lower']   = streets_sel[col_list].multiply(streets_sel['scale_lower'], axis=0).sum(axis=1)
streets_sel['upper']   = streets_sel[col_list].multiply(streets_sel['scale_upper'], axis=0).sum(axis=1)
streets_sel['central'] = streets_sel[col_list].sum(axis=1)

ax.plot(streets_sel['Year'], streets_sel['central'], 
         #marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.5,
         color='k',
         label='Streets et al. (2019)')

ax.fill_between(streets_sel['Year'], streets_sel['lower'], streets_sel['upper'], 
                facecolor='gainsboro')
# -- plot Hylander and Meili (2002)
ax.plot(hylander['Year'], hylander['Globally mined [Mg/yr]'], 
         color='steelblue', label='Hylander and Meili (2003)');
# -- plot Guerrero and Schneider (2023)
ax.plot(guerrero['Year'], guerrero['Hg SP [Mg/yr]'], 
         color='firebrick', label='Guerrero and Schneider (2023)');
# -- add labels
ax.set_xlabel('Year')
ax.set_ylabel('Hg Releases [Mg a$^{-1}$]$^{*}$')
#ax.set_title('Comparison of Global Hg Inventories')

# remove grid 
# set major ticks
ax.set_xticks(np.arange(1500, 2021, 100))
ax.set_xticks(np.arange(1500, 2021, 20), minor=True);
# add grid for major ticks
ax.grid(which='major', color='0.2', linestyle='-', linewidth=0.5)
# add grid for minor ticks
#ax.grid(which='minor', color='0.8', linestyle='-', linewidth=0.5)
ax.set_xlim(1500, 2020)

modify_ticks(ax)
# add ticks to the right and top
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

#ax.legend(loc='center left', fontsize=8, frameon=False)

# add annotation
# -- 1510 - 2000
y1, y2 = 1510, 2000
hylander_1510_2000        = hylander[(hylander['Year']>=y1) & (hylander['Year']<=y2)]['Globally mined [Mg/yr]'].sum()*1e-3
streets_1510_2000_central = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['central'], include_last_year=True, include_first_year=True).sum()
streets_1510_2000_lower   = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['lower'], include_last_year=True, include_first_year=True).sum()
streets_1510_2000_upper   = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['upper'], include_last_year=True, include_first_year=True).sum()

# -- 1510 - 1900
y1, y2 = 1510, 1900
hylander_1510_1900        = hylander[(hylander['Year']>=y1) & (hylander['Year']<=y2)]['Globally mined [Mg/yr]'].sum()*1e-3
streets_1510_1900_central = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['central'], include_last_year=True, include_first_year=True).sum()
streets_1510_1900_lower   = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['lower'], include_last_year=True, include_first_year=True).sum()
streets_1510_1900_upper   = get_cumulative_emissions(streets_sel[(streets_sel['Year']>=y1) & (streets_sel['Year']<=y2)], ['upper'], include_last_year=True, include_first_year=True).sum()
guerrero_1510_1900        = guerrero[(guerrero['Year']>=y1) & (guerrero['Year']<=y2)]['Hg SP [Mg/yr]'].sum()*1e-3

# --- 
fontsize = 7.5

ax.annotate('', xy=(2000, 18500), xytext=(1510, 18500),
            arrowprops=dict(arrowstyle='->', color='0.2', lw=0.5),
            annotation_clip=False)
# add tiny vertical line at 2000 with axis transformation
ax.vlines(2000, 18000, 19000, transform=ax.transData, color='0.2', linestyle='-', linewidth=0.5)

ax.annotate('  Cumulative Releases (1510 - 2000)', xy=(1510, 18540), xytext=(1510, 18540), ha='left', va='bottom', color='0.2', fontsize=fontsize)
# annotate cumulative from Streets et al. 2019
ax.text(1510, 18000, f'  (1) {streets_1510_2000_central:.0f} [{streets_1510_2000_lower:.0f} - {streets_1510_2000_upper:.0f}] Gg',
        ha='left', va='top', fontsize=fontsize, color='k')
# annotate cumulative from Hylander and Meili 2002
ax.text(1510, 17000, f'  (2) {hylander_1510_2000:.0f} Gg',
        ha='left', va='top', fontsize=fontsize, color='steelblue')

# add horizontal line with arrow on end from 1510 to 1900
ax.annotate('', xy=(1900, 14000), xytext=(1510, 14000),
            arrowprops=dict(arrowstyle='->', color='0.2', lw=0.5),
            annotation_clip=False)
# add tiny vertical line at 1900 with axis transformation
ax.vlines(1900, 13500, 14500, transform=ax.transData, color='0.2', linestyle='-', linewidth=0.5)
ax.annotate('  Cumulative Releases (1510 - 1900)', xy=(1510, 14040), xytext=(1510, 14040), ha='left', va='bottom', color='0.2', fontsize=8)
ax.text(1510, 13500, f'  (1) {streets_1510_1900_central:.0f} [{streets_1510_1900_lower:.0f} - {streets_1510_1900_upper:.0f}] Gg', ha='left', va='top', fontsize=fontsize, color='k')
ax.text(1510, 12500, f'  (2) {hylander_1510_1900:.0f} Gg', ha='left', va='top', fontsize=fontsize, color='steelblue')
ax.text(1510, 11500, f'  (3) {guerrero_1510_1900:.0f} Gg', ha='left', va='top', fontsize=fontsize, color='firebrick')
# add legend as annotation
ax.text(1510, 7200, '  Legend', ha='left', va='bottom', fontsize=fontsize, color='0.2', zorder=4)
ax.text(1510, 6000, '  (1) Streets et al. (2019)', ha='left', va='bottom', fontsize=fontsize, color='k', zorder=4)
ax.text(1510, 5000, '  (2) Hylander and Meili (2003)', ha='left', va='bottom', fontsize=fontsize, color='steelblue', zorder=4)
ax.text(1510, 4000, '  (3) Guerrero and Schneider (2023)', ha='left', va='bottom', fontsize=fontsize, color='firebrick', zorder=4)
# put white background behind legend
ax.fill_between([1510, 1700], 3600, 8500, color='0.98', edgecolor='0.7', linewidth=0.5, zorder=3)

ax.set_ylim(-100, 20000)
ax.grid(False)

plt.savefig('../figures/figure_S4.png', dpi=1200, bbox_inches='tight')