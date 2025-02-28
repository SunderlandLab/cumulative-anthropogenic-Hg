import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acgc import figstyle
import json

design_dict = json.load(open('../inputs/meta/design_dict.json', 'r'))

df = pd.read_csv('../output/main/constraint_table_sector_SSP1-26_1510_2300.csv')

constraints = {'Atmospheric Burden (Gg)': {'min': 3.9, 'max': 4.5,  'central': 4.3,   'credible_range': [3.9, 4.5],     'units':'Gg', 'title': 'Atmosphere', 'source':'', 'bool_quantiles':False},
               'Preind atm. EF':          {'min': None, 'max': None,  'central': 2.674, 'credible_range': [1.998, 3.654], 'units':'EF', 'title': 'EF$_{\mathrm{preind}}$', 'source':'', 'bool_quantiles':True},
               'Alltime atm. EF':         {'min': None, 'max': None, 'central': 7.789,  'credible_range': [3.194, 10.50], 'units':'EF', 'title': 'EF$_{\mathrm{alltime}}$','source':'', 'bool_quantiles':True},
               'Upper Ocean Conc. (pM)': {'min': 0.61, 'max': 1.50, 'central': 0.94, 'credible_range': [0.78, 1.2], 'units': 'pmol L$^{-1}$', 'title': 'Upper Ocean ($<1500$ m)', 'source':'this work', 'bool_quantiles':True},
               'Deep Ocean Conc. (pM)':  {'min': 0.83, 'max': 1.60, 'central': 1.2,  'credible_range': [1.0, 1.3],'units': 'pmol L$^{-1}$', 'title': 'Deep Ocean ($>1500$ m)', 'source':'this work', 'bool_quantiles':True},
}

reference_fn = '../output/main/constraint_table_sector_SSP1-26_1510_2300.csv'

dir = '../output/sensitivity/'
ext = 'constraint_table_sector_SSP1-26_1510_2010.csv'

fn_list_sensitivity = []
# -- sensitivity analysis [emission magnitude]
for subdir in ['streets_high', 'streets_low']:
    fn_list_sensitivity.append(dir + subdir + '/' + ext)

def modify_ticks(ax):
    c = '0.3'
    # set spine and tick color
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
    # set tick color 
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    return

def plot_axis(ax, df, constraint_name:str, orientation='horizontal', bar_bounds=[0.1, 0.9], lw=1, color='tab:blue', alpha1=0.5, alpha2=0.5, **kwargs):
    ''' plot the axis for a given constraint'''
    df = df[constraint_name]
    if orientation == 'horizontal':
        # vertical box showing the range of values
        ax.axhspan(ymin=df['min'], ymax=df['max'], xmin=bar_bounds[0], xmax=bar_bounds[1], facecolor=color, edgecolor=color, lw=0, alpha=alpha2)
        ax.axhspan(ymin=df['credible_range'][0], ymax=df['credible_range'][1], xmin=bar_bounds[0], xmax=bar_bounds[1], facecolor=color, edgecolor=color, lw=0, alpha=alpha1)
        ax.set_ylabel(df['units']) # add unit label

    elif orientation == 'vertical':
        # horizontal box showing the range of values
        ax.axvspan(xmin=df['min'], xmax=df['max'], ymin=bar_bounds[0], ymax=bar_bounds[1], facecolor=color, edgecolor=color, lw=0, alpha=alpha2)
        ax.axvspan(xmin=df['credible_range'][0], xmax=df['credible_range'][1], ymin=bar_bounds[0], ymax=bar_bounds[1], facecolor=color, edgecolor=color, lw=0, alpha=alpha1)
        ax.set_xlabel(df['units']) # add unit label

    return ax

def plot_individual_sensitivity_point(ax, constraint_name, fn: str, orientation='horizontal', pos=0.45, **kwargs):
    ''' plot the sensitivity point for a given constraint'''
    df = pd.read_csv(fn)
    df = df[df['Constraint Name']==constraint_name]
    
    if orientation == 'vertical':
        ax.scatter(x=df['Model'], y=pos, **kwargs)
        
    elif orientation == 'horizontal':
        ax.scatter(x=pos, y=df['Model'], **kwargs)
    return ax

def plot_model_bounds(ax, constraint_name, fn_list, orientation='horizontal', pos=0.45, **kwargs):
    lw = 2

    constraint_list = []
    for fn in fn_list:
        tmp = pd.read_csv(fn)
        constraint_list.append(tmp[tmp['Constraint Name']==constraint_name]['Model'])

    if orientation == 'vertical':
        ax.hlines(y=pos, xmin=np.min(constraint_list), xmax=np.max(constraint_list), lw=lw, **kwargs)
        # add vertical fliers for the model bounds
        width = 0.06
        ax.vlines(x=np.min(constraint_list), ymin=pos-width, ymax=pos+width, lw=lw*0.75, **kwargs)
        ax.vlines(x=np.max(constraint_list), ymin=pos-width, ymax=pos+width, lw=lw*0.75, **kwargs)

    elif orientation == 'horizontal':
        ax.vlines(x=pos, ymin=np.min(constraint_list), ymax=np.max(constraint_list), **kwargs)

    return ax

def overplot_out_of_bounds(ax, df, constraint_name, fn_list, orientation='horizontal', pos=0.45, **kwargs):
    lw = 2
    width = 0.06

    df = df[constraint_name]
    constraint_list = []
    for fn in fn_list:
        tmp = pd.read_csv(fn)
        constraint_list.append(tmp[tmp['Constraint Name']==constraint_name]['Model'])

    if orientation == 'vertical':
        # add vertical fliers for the model bounds
        if df['max'] != None:
            if df['max'] < np.max(constraint_list):
                ax.hlines(y=pos, xmin=df['max'], xmax=np.max(constraint_list), lw=lw, zorder=5, **kwargs)
                ax.vlines(x=np.max(constraint_list), ymin=pos-width, ymax=pos+width, lw=lw*0.75, zorder=5, **kwargs)
        if df['min'] != None:
            if df['min'] > np.min(constraint_list):
                ax.hlines(y=pos, xmin=np.min(constraint_list), xmax=df['min'], lw=lw, zorder=5, **kwargs)
                ax.vlines(x=np.min(constraint_list), ymin=pos-width, ymax=pos+width, lw=lw*0.75, zorder=5, **kwargs)

plt_dict = {
    'Atmosphere':{
        'Atmospheric Burden (Gg)': {'bar_bounds':[0.08, 0.68], 'axis_lims': [3, 6],  'axis_ticks': [3, 4, 5, 6], 'axis_ticklabels': [3, 4, 5, 6], 'title': 'Atmosphere', 'annotation': 'Tropospheric Reservoir (2010)', 'units':'Gg', 'detail': 'stations\n$n$=40'},
     },
     'Sediment Cores': {
        'Preind atm. EF': {'bar_bounds': [0.08, 0.68], 'axis_lims': [0, 11],  'axis_ticks': None, 'axis_ticklabels': None, 'title': 'Sediment Cores', 'annotation':'Pre-industrial to 20$^\mathrm{th}$C$_\mathrm{max}$ Enrichment Factor', 'units':' EF (unitless)', 'detail': 'observations\n$n$=93 (NH)\n$n$=19 (SH)'}, # 'EF$_{\mathrm{pre\u2013ind}}$' #'EF: Pre-ind. to 20$\mathrm{^{th}}$C max'
        'Alltime atm. EF': {'bar_bounds':[0.08, 0.68], 'axis_lims': [0, 11], 'axis_ticks': None, 'axis_ticklabels': None, 'title': '', 'annotation':'Natural to 20$^\mathrm{th}$C$_\mathrm{max}$ Enrichment Factor', 'units':'EF (unitles)', 'detail': 'observations\n$n$=5 (NH)\n$n$=10 (SH)'},
     },
     'Seawater': {
        'Upper Ocean Conc. (pM)':  {'bar_bounds': [0.08, 0.68], 'axis_lims': [0.2, 1.8],  'axis_ticks': [0.2, 0.6, 1.0, 1.4, 1.8], 'axis_ticklabels': [0.2, 0.6, 1.0, 1.4, 1.8], 'title': 'Seawater', 'annotation':'Upper Ocean (0-1500 m)', 'units':'pmol L$^{-1}$', 'detail': 'stations\n$n$=382\nsamples\n$n$=4481'}, #'title':'Upper Ocean\n(0-1500 m)'},
        'Deep Ocean Conc. (pM)':   {'bar_bounds': [0.08, 0.68], 'axis_lims': [0.2, 1.8],  'axis_ticks': [0.2, 0.6, 1.0, 1.4, 1.8], 'axis_ticklabels': [0.2, 0.6, 1.0, 1.4, 1.8], 'title': '', 'annotation':'Deep Ocean (\u003E1500 m)', 'units':'pmol L$^{-1}$', 'detail': 'stations\n$n$=159\nsamples\n$n$=1240'},#'5%: 0.6\n25%: 0.6\n50%: 0.9\n75%: 1.1\n95%: 1.4'}, #'title':'Deep Ocean\n(>1500 m)'},
    },
}

fn = reference_fn

fig = plt.figure(figsize=(3.4, 7))
gs  = fig.add_gridspec(6, 1,)
orientation = 'vertical'
bar_bounds = [0.1, 0.7]
pos = np.mean(bar_bounds)

fig = plt.figure(figsize=(3.4, 7), tight_layout=True)
# Set ratios for main panels (ax panels)
gs = fig.add_gridspec(3, 1, hspace=0.5, height_ratios=[1, 2, 2])

for i, constraint_type in enumerate(plt_dict.keys()):
    
    dict_sel = plt_dict[constraint_type]
    n_constraints = len(dict_sel.keys())

    # Main panel (ax)
    ax = fig.add_subplot(gs[i, 0])
    ax.set_title(constraint_type, fontsize=12, color='0.1')
    # add letter label 
    # get height of the title
    title_height = ax.title.get_position()[1]
    # add letter label at the top left corner at same height as title
    ax.text(x=0, y=1.02, s=chr(97+i), fontsize=14, color='0.1', va='bottom', ha='left', transform=ax.transAxes, fontweight='bold')
    
    # Customize main panel (remove spines and ticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_yticks([])
    ax.set_xticks([])

    # Create subpanels (subax) within each main panel
    gs_sub = gs[i].subgridspec(n_constraints, 1, wspace=0.0, hspace=0.0, height_ratios=[1]*n_constraints)

    for j, constraint_name in enumerate(dict_sel.keys()):
        
        subax = fig.add_subplot(gs_sub[j, 0])
        bar_bounds = dict_sel[constraint_name]['bar_bounds']
        pos = np.mean(bar_bounds)
        
        # -- Example plot function calls (replace these with your actual plotting functions)
        subax = plot_axis(subax, constraints, constraint_name, orientation=orientation, bar_bounds=bar_bounds, color="#0072BD", alpha1=1, alpha2=0.4)
        # -- plot model central value
        subax = plot_individual_sensitivity_point(subax, constraint_name=constraint_name, fn=fn, pos=pos, orientation=orientation, facecolor="#EDB120", edgecolor='whitesmoke', marker='o', s=100, zorder=3)
        # -- plot model bounds
        subax = plot_model_bounds(subax, constraint_name=constraint_name, fn_list=fn_list_sensitivity, orientation=orientation, pos=pos, color="#EDB120")
        
        subax.set_xlim(dict_sel[constraint_name]['axis_lims'])
        subax.set_ylim(0, 1)
        subax.set_yticks([])

        # -- add annotation
        y = bar_bounds[1] + 0.08
        subax.text(x=0.5, y=y, s=dict_sel[constraint_name]['annotation'], ha='center', va='bottom', transform=subax.transAxes, fontsize=9, color='0.2', fontweight='light')
        
        subax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        subax.grid(False)
        # make subpanel background transparent
        subax.patch.set_alpha(0)
        # remove x-axis label if not the last subpanel
        if j < n_constraints-1:
            subax.set_xlabel('')
            # remove bottom spine
            subax.spines['bottom'].set_visible(False)
            # remove xticks
            subax.set_xticks([])
            # remove xtick labels
            subax.set_xticklabels([])
            
        # remove top spine if not the first subpanel
        if j > 0:
            subax.spines['top'].set_visible(False)
            
        # if j is the last subpanel, add x-axis label
        if j == n_constraints-1:
            subax.set_xlabel(dict_sel[constraint_name]['units'], fontsize=8, color='0.3')
            # set xticks based on axis limits
            if dict_sel[constraint_name]['axis_ticks'] is not None:
                subax.set_xticks(dict_sel[constraint_name]['axis_ticks'])
                subax.set_xticklabels(dict_sel[constraint_name]['axis_ticklabels'])
            # make minor and major ticks visible
            subax.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=8)
            # annotate quantiles if the range is based on quantiles

        # annotate quantiles if the range is based on quantiles
        if constraints[constraint_name]['bool_quantiles']:
            # check to see if span of credible range is less than 30% of the xaxis span
            xmin, xmax = subax.get_xlim()
            constraint_min, constraint_max = constraints[constraint_name]['credible_range']
            if (constraint_max - constraint_min) > 0.2*(xmax - xmin):
                subax.text(x=constraints[constraint_name]['credible_range'][0], y=0.16, s=' 25%', ha='left', va='center', fontsize=7, color='1.0')
                subax.text(x=constraints[constraint_name]['credible_range'][1], y=0.16, s='75% ', ha='right', va='center', fontsize=7, color='1.0')
            else:
                print(f'suppressing quantile annotation for {constraint_name} because range is too small')

            if constraints[constraint_name]['min'] != None:
                # suppress annotation for deep ocean because it's too close to the 25% quantile
                #if constraint_name != 'Deep Ocean Conc. (pM)':
                subax.text(x=constraints[constraint_name]['min'], y=0.16, s=' 5%', ha='left', va='center', fontsize=7, color='0.')
            if constraints[constraint_name]['max'] != None:
                subax.text(x=constraints[constraint_name]['max'], y=0.16, s='95% ', ha='right', va='center', fontsize=7, color='0.')

        # add detail text annotation if available
        if dict_sel[constraint_name]['detail'] != '':
            default_x = 0.02
            default_y = pos
            if constraint_name == 'Preind atm. EF':
                default_x = 0.4
            subax.text(x=default_x, y=default_y, s=dict_sel[constraint_name]['detail'], ha='left', va='center', transform=subax.transAxes, fontsize=7, c='0.4')

        modify_ticks(subax)

    ax.grid(False)

plt.savefig(f'../figures/figure_2.pdf', format='pdf')
plt.savefig(f'../figures/figure_2.png', dpi=1200, format='png')