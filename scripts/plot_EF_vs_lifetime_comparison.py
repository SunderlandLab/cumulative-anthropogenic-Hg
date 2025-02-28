import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acgc import figstyle
from helpers import *

reservoirs_non_waste = ['atm', 'ocs', 'oci', 'ocd', 'tf', 'ts', 'ta']
EFs = pd.read_csv('../output/main/tables/mass_table_sector_SSP1-26_2010.csv')
EFs = EFs[['reservoir','EF_alltime']]

update_names = {
    'Atmosphere': 'atm', 
    'Terrestrial Fast': 'tf', 'Terrestrial Slow': 'ts', 'Terrestrial Protected': 'ta',
    'Ocean Surface': 'ocs', 'Ocean Intermediate': 'oci', 'Ocean Deep': 'ocd',}

EFs['reservoir'] = EFs['reservoir'].replace(update_names)

rate_matrix = pd.read_csv('../output/main/rate_matrix_sector_SSP1-26_1510_2300.csv')

def get_lifetime(rate_matrix, compartment='atm'):
    col_index = rate_matrix.columns.get_loc(compartment)
    tau = 1/np.abs(rate_matrix[compartment].values[col_index])
    return tau

reservoirs = list(rate_matrix.columns)
lifetimes  = [get_lifetime(rate_matrix, compartment=comp) for comp in reservoirs_non_waste]
enrichments = [EFs[EFs['reservoir']==comp]['EF_alltime'].values[0] for comp in reservoirs_non_waste]

plt.figure(figsize=(6, 4))
plt.scatter(lifetimes, enrichments, edgecolor='0.1', s=100, zorder=4)
plt.xscale('log')
labels = {
    'atm': 'atmosphere', 
    'ocs': 'surface\nocean', 'oci': 'intermediate\nocean', 'ocd': 'deep\nocean',
    'tf':'fast\nterrestrial', 'ts':'slow\nterrestrial', 'ta':'protected\nterrestrial',}

# label each point with the name of the compartment
for i, txt in enumerate(reservoirs_non_waste):
    if txt in labels:
        #plt.annotate(labels[txt], (lifetimes[i]*1.5, enrichments[i]-0.1), fontsize=9, color='0.3', zorder=3)
        # add background to text with alpha=0.5
        plt.annotate(labels[txt], (lifetimes[i]*1.5, enrichments[i]-0.2), fontsize=9, color='0.3', 
        backgroundcolor=(1, 1, 1, 0.5), zorder=2, ha='left', va='bottom')

plt.xlim(1e-1, 3e5)
plt.ylim(0.5, 8.5)
# -
c = '0.1'
plt.xlabel('Lifetime (years)', color=c)
plt.ylabel('All-time Enrichment Factor', color=c)
# make axis spines grey
plt.gca().spines['bottom'].set_color(c)
plt.gca().spines['top'].set_color(c)
plt.gca().spines['right'].set_color(c)
plt.gca().spines['left'].set_color(c)
# make minor and major ticks grey
plt.tick_params(axis='x', colors=c)
plt.tick_params(axis='y', colors=c)
plt.tick_params(which='minor', colors=c)
plt.tick_params(which='major', colors=c)

# -- add annotation in lower left to define EF
#plt.text(0.05, 0.05, 'All-time Enrichment Factor = Mass (2010) / Mass (natural steady state)', transform=plt.gca().transAxes, fontsize=9, color='0.3')

plt.savefig('../figures/figure_S9.pdf', bbox_inches='tight')
plt.savefig('../figures/figure_S9.png', dpi=1200, bbox_inches='tight')
