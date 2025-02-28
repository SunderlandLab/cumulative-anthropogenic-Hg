import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

from acgc import figstyle

def modify_ticks(ax):
    c = '0.2'
    # set spine and tick color
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
    # set tick color 
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    # set color of axis labels
    ax.yaxis.label.set_color(c)
    ax.xaxis.label.set_color(c)
    return

# Read in the compilation file
df = pd.read_csv('../output/sensitivity/contour_plot/constraint_table_compilation_for_contour.csv')

df_out = pd.DataFrame(columns=list(df.columns))
# loop over w_wf values
for w_wf in df['w_wf'].unique():
    # loop over w_ws values
    for w_ws in df['w_ws'].unique():
        # select the subset of the dataframe that matches the current w_wf, w_ws, and w_wa values
        sel = df.loc[(df['w_wf'] == w_wf) & (df['w_ws'] == w_ws)].copy()
        # if the subset dataframe is empty, skip to the next iteration
        if sel.empty:
            sel = pd.DataFrame({'w_wf': [w_wf], 'w_ws': [w_ws], 'w_wa': [np.nan], 
                                'Atmospheric Burden (Gg)':[np.nan],
                                #'Peak modern atm. EF': [np.nan],
                                'Preind atm. EF': [np.nan],
                                'Alltime atm. EF': [np.nan],
                                'Upper Ocean Conc. (pM)':[np.nan],
                                'Deep Ocean Conc. (pM)':[np.nan]})
        # if the subset dataframe is not empty, make a contour plot
        df_out = pd.concat([df_out, sel], ignore_index=True)

df_out = df_out.sort_values(by=['w_wf', 'w_ws'])
df_out = df_out.reset_index(drop=True)

df_out['Cartoon'] = 2*df_out['w_wf'] + 1.25*df_out['w_ws'] + 0.1*df_out['w_wa']

# from: https://stackoverflow.com/questions/55665167/asymmetric-color-bar-with-fair-diverging-color-map
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# define function to grid the data for a given variable
def get_xyz(df, column_name):
    # assert that matrix is square
    assert len(df['w_wf'].unique()) == len(df['w_ws'].unique())
    X, Y = np.meshgrid(df['w_wf'].unique(), df['w_ws'].unique())
    # reshape df[column_name] into a matrix
    Z = df[column_name].values.reshape(X.shape)

    return X, Y, Z

# now make triangular contour plots
def plot_contour(ax, X, Y, Z, title='', levels=10, cbar_label='', cbar_shrink=1.0, cmap_midpoint=0, cbar_ticks=None, cmap='viridis'):
    cmap = plt.get_cmap(cmap)
    cmap.set_under('whitesmoke')
    cmap.set_over('whitesmoke')
    # plot the contour
    cs = ax.contourf(X, Y, Z, cmap=cmap, levels=levels, norm=MidpointNormalize(midpoint=cmap_midpoint))
    # add a title
    ax.set_title(title, fontsize=8)
    # set the x and y limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # set x and y ticks to have the same number of ticks
    ax.set_xticks(np.arange(0,1.1,0.2), minor=False)
    ax.set_yticks(np.arange(0,1.1,0.2), minor=False)
    # set minor ticks
    ax.set_xticks(np.arange(0,1.1,0.1), minor=True)
    ax.set_yticks(np.arange(0,1.1,0.1), minor=True)
    # set the x and y labels
    ax.set_xlabel('slow waste fraction')
    ax.set_ylabel('fast waste fraction')
    # add a colorbar
    #cbar = fig.colorbar(cs, ax=ax, ticks=cbar_ticks, label=cbar_label, shrink=cbar_shrink)
    # -- place colorbar as inset -- 
    # (replaces the above line)
    # from https://stackoverflow.com/questions/18211967/position-colorbar-inside-figure
    #fig.tight_layout()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbaxes = inset_axes(ax, width="50%", height="10%", loc='upper left',
                   bbox_to_anchor=(0.4,0.1,0.95,0.7), bbox_transform=ax.transAxes) 
    fig.colorbar(cs, cax=cbaxes, ticks=cbar_ticks, orientation='horizontal', label=cbar_label)
    # set fontsize to 8 for colorbar labels and ticklabels
    cbaxes.xaxis.label.set_size(8)
    cbaxes.tick_params(labelsize=8)

def add_specific_contour(ax, X, Y, Z, value, color='k', linestyle='-', linewidth=1):
    # add a specific contour
    cs = ax.contour(X, Y, Z, levels=[value], colors=color, 
                    linestyles=linestyle, linewidths=linewidth, zorder=10)

def plot_polygon(ax, x, y, color='k', linestyle='-', linewidth=1):
    # plot a polygon
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=10)

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_bounding_lines(ax, c='grey', zorder=20, **kwargs):
    ax.plot([0, 0], [1, 1], color=c, zorder=zorder, **kwargs)
    ax.plot([1, 1], [0, 0], color=c, zorder=zorder, **kwargs)
    ax.plot([0, 1], [1, 0], color=c, zorder=zorder, **kwargs)

# set axis parameters
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'axes.linewidth': 1})
# change spine width to 1
plt.rcParams['axes.linewidth'] = 1

inset_title_fontsize = 10

from matplotlib.colors import ListedColormap
cmap = plt.colormaps.get_cmap('RdBu_r')
# trim cmap by removing the first and last 20% of the colors
cmap = cmap(np.linspace(0.2, 0.8, 256))
cmap = ListedColormap(cmap)

obs_line_color = 'k'
bounding_line_color = 'k'
# set up the figure
fig = plt.figure(figsize=(7.9, 5.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, wspace=0.1, hspace=0.2, 
                      width_ratios=[1, 1, 1], height_ratios=[1, 1])

subplot_settings = {'ax1': {'gs_row_idx':0, 'gs_col_idx':0, 'col_name': 'Atmospheric Burden (Gg)', 
                            'title': 'Reservoir Mass (2010)',#'Atmosphere', 
                            'cbar_label':'Gg',
                            'levels':np.arange(3.8, 4.81, 0.01), 'cmap_midpoint':4.25, 'cbar_ticks':[3.5, 4, 4.5, 5], 'cmap':cmap,
                            'constraint_mid':4.25, 'constraint_upper':4.5, 'constraint_lower':4.0},

                    'ax2': {'gs_row_idx':0, 'gs_col_idx':1, 'col_name': 'Preind atm. EF', 
                            'title': 'Pre-industrial to 20$^{\mathrm{th}}$C$\mathrm{_{max}}$', 
                            'cbar_label':'EF',
                            'levels':np.arange(1.60, 3.72, 0.01), 'cmap_midpoint':2.7, 'cbar_ticks':[2.5, 3.5], 'cmap':cmap,
                            'constraint_mid':2.7, 'constraint_upper':3.72, 'constraint_lower':1.60},

                    'ax3': {'gs_row_idx':0, 'gs_col_idx':2, 'col_name': 'Alltime atm. EF', 
                            'title': '',#'Deposition Enrichment Factor\n(Natural - 20thC Max)', 
                            'cbar_label':'EF',
                            'levels':np.arange(2.15, 14.43, 0.01), 'cmap_midpoint':8.3, 'cbar_ticks':[4, 8, 12], 'cmap':cmap,
                            'constraint_mid':8.3, 'constraint_upper':14.43, 'constraint_lower':2.15},                    

                    'ax4': {'gs_row_idx':1, 'gs_col_idx':0, 'col_name': 'Upper Ocean Conc. (pM)', 
                            'title': '',#'Upper Ocean\n(<1500 m)', 
                            'cbar_label':'pM',
                            'levels':np.arange(0.61, 1.51, 0.01), 'cmap_midpoint':0.94, 'cbar_ticks':[0.7, 1.1, 1.5], 'cmap':cmap,
                            'constraint_mid':0.94, 'constraint_upper':1.2, 'constraint_lower':0.78},

                    'ax5': {'gs_row_idx':1, 'gs_col_idx':1, 'col_name': 'Deep Ocean Conc. (pM)', 
                            'title':'',#'Deep Ocean\n(>1500 m)', 
                            'cbar_label':'pM',
                            'levels':np.arange(0.83, 1.61, 0.01), 'cmap_midpoint':1.2, 'cbar_ticks':[0.8, 1, 1.2, 1.4, 1.6], 'cmap':cmap,
                            'constraint_mid':1.2, 'constraint_upper':1.3, 'constraint_lower':0.87},
                    }

for k, plt_dict in subplot_settings.items():
    X, Y, Z = get_xyz(df_out, plt_dict['col_name'])
    ax = fig.add_subplot(gs[plt_dict['gs_row_idx'], plt_dict['gs_col_idx']])
    plot_contour(ax, X, Y, Z, title=None, cbar_label=plt_dict['cbar_label'], 
                 cbar_shrink=0.6, cmap=plt_dict['cmap'], levels=plt_dict['levels'],
                 cmap_midpoint=plt_dict['cmap_midpoint'], cbar_ticks=plt_dict['cbar_ticks'])
    for val, ls in zip(['constraint_mid', 'constraint_lower','constraint_upper'], ['--', '-', '-']):
        add_specific_contour(ax, X, Y, Z, plt_dict[val], color=obs_line_color, linestyle=ls, linewidth=0.8)
    ax.set_facecolor('none') # remove ax background color
    remove_spines(ax)
    add_bounding_lines(ax, c=bounding_line_color, linewidth=1, zorder=10)
    # add background patch to display in constraint-incompatible region
    constraint_incompatible_color = '0.15' #'#fffff8'
    ax.fill([0, 1, 0, 0], [0, 0, 1, 0], color=constraint_incompatible_color, alpha=1, zorder=0)
    ax.set_xticks(np.arange(0,1.1,0.1), minor=True)
    ax.set_yticks(np.arange(0,1.1,0.1), minor=True)
    ax.grid(False)
    modify_ticks(ax)

from matplotlib.patches import Rectangle

# Add bounding boxes around specified subplot groups
def add_subplot_border(fig, axs, color='black', lw=2):
    # Get the bounding box of the subplots as a combined area
    bbox = axs[0].get_position()
    for ax in axs[1:]:
        bbox = bbox.union(ax.get_position())
    
    # Add a rectangle to the figure using the bounding box coordinates
    rect = Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height,
                     linewidth=lw, edgecolor=color, facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)

# create box in fig
rect1 = Rectangle((0.0, 0.53), 0.32, 0.47, linewidth=0.3, edgecolor='0.9', facecolor='0.98', transform=fig.transFigure, zorder=-1)
fig.patches.append(rect1)
# place letter in upper left above box
fig.text(0.0, 1.005, 'a', fontsize=14, fontweight='bold', ha='left', va='bottom', transform=fig.transFigure)
fig.text(0.16, 1.005, 'Atmosphere', fontsize=14, ha='center', va='bottom', transform=fig.transFigure)
fig.text(0.31, 0.98, 'Reservoir Mass (2010)', fontsize=10, ha='right', va='top', transform=fig.transFigure)

rect2 = Rectangle((0.34, 0.53), 0.66, 0.47, linewidth=0.3, edgecolor='0.9', facecolor='0.98', transform=fig.transFigure, zorder=-1)
fig.patches.append(rect2)
# place letter in upper left above box
fig.text(0.34, 1.005, 'b', fontsize=14, fontweight='bold', ha='left', va='bottom', transform=fig.transFigure)
fig.text(0.67, 1.005, 'Lake Sediment Enrichment Factors', fontsize=14, ha='center', va='bottom', transform=fig.transFigure)
fig.text(0.66, 0.98, 'Pre-industrial to 20$^{\mathrm{th}}$C$\mathrm{_{max}}$', fontsize=10, ha='right', va='top', transform=fig.transFigure)
fig.text(0.99, 0.98, 'Natural to 20$^{\mathrm{th}}$C$\mathrm{_{max}}$', fontsize=10, ha='right', va='top', transform=fig.transFigure)

rect3 = Rectangle((0.0, 0.0), 0.66, 0.47, linewidth=0.3, edgecolor='0.9', facecolor='0.98', transform=fig.transFigure, zorder=-1)
fig.patches.append(rect3)
fig.text(0.0, 0.475, 'c', fontsize=14, fontweight='bold', ha='left', va='bottom', transform=fig.transFigure)
fig.text(0.33, 0.475, 'Seawater Concentrations', fontsize=14, ha='center', va='bottom', transform=fig.transFigure)
fig.text(0.32, 0.435, 'Upper Ocean (0 - 1500 m)', fontsize=10, ha='right', va='top', transform=fig.transFigure)
fig.text(0.65, 0.435, 'Deep Ocean (>1500 m)', fontsize=10, ha='right', va='top', transform=fig.transFigure)

# get first subplot
ax1 = fig.get_axes()[0]
#    #annotate upper left corner to indicate that the constraint is not applicable
ax1.text(s='incompatible with observations', x=0.06, y=0.85, 
        ha='left', va='top', fontsize=10, color='w', transform=ax1.transAxes,
        rotation=-45)

#plt.savefig('../figures/LW_constraints_contour_plot.pdf', bbox_inches='tight')
plt.savefig('../figures/LW_constraints_contour_plot.png', bbox_inches='tight', dpi=1200)