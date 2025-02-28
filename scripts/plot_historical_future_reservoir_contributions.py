import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from acgc import figstyle

def make_plot(ax, scenario='SSP1-26', reservoir=['atm'], xlim=[1850, 2300], style_dict=None):
    """
    Generates a stacked fill_between plot showing the contributions
    from Natural, pre-2010, and post-2010 for a given scenario/reservoir.
    """

    if style_dict is None:
        style_dict = {
            'color0': 'k', 'alpha0': 0.5,
            'color1': 'tab:blue', 'alpha1': 0.8,
            'color2': 'tab:red', 'alpha2': 1.0,
            'color3': 'tab:green', 'alpha3': 0.8,
            'color4': 'tab:orange', 'alpha4': 1.0,
        }

    # Read data
    # --------------------------------------------
    natural  = pd.read_csv(f'../output/main/time_attribution/past_and_future/{scenario}/all_inputs_output_sector_{scenario}_1509_1510.csv')
    pre_2010 = pd.read_csv(f'../output/main/time_attribution/past_and_future/{scenario}/fragments_sector_{scenario}_1510_2010.csv').groupby(by=['media', 'Year'], as_index=False).sum(numeric_only=True)
    pre_2010_LW = pre_2010[pre_2010['media']=='LW'].copy()
    pre_2010_Air = pre_2010[pre_2010['media']=='Air'].copy()
    post_2010 = pd.read_csv(f'../output/main/time_attribution/past_and_future/{scenario}/fragments_sector_{scenario}_2010_2300.csv').groupby(by=['media', 'Year'], as_index=False).sum(numeric_only=True)
    post_2010_LW = post_2010[post_2010['media']=='LW'].copy()
    post_2010_Air = post_2010[post_2010['media']=='Air'].copy()

    time = natural['Year'].values

    # Convert from Mg to Gg (divide by 1e3)
    # --------------------------------------------
    m_natural = natural[reservoir].sum(axis=1).values[0] / 1e3
    m_natural_array = [m_natural]*len(time)  # horizontal line (repeated)
    m_pre_2010_LW  = pre_2010_LW[reservoir].sum(axis=1).values / 1e3
    m_pre_2010_Air = pre_2010_Air[reservoir].sum(axis=1).values / 1e3
    m_post_2010_LW = post_2010_LW[reservoir].sum(axis=1).values / 1e3
    m_post_2010_Air = post_2010_Air[reservoir].sum(axis=1).values / 1e3

    # stack_components for plotting
    # --------------------------------------------
    m1 = m_natural_array
    m2 = m1 + m_pre_2010_LW
    m3 = m2 + m_pre_2010_Air
    m4 = m3 + m_post_2010_LW
    m5 = m4 + m_post_2010_Air

    # Plot the stacked fill
    # --------------------------------------------
    f0 = ax.fill_between(
        time, 0, m1,
        color=style_dict['color0'], alpha=style_dict['alpha0'],
        label='Natural', zorder=1
    )
    f1 = ax.fill_between(
        time, m1, m2,
        color=style_dict['color1'], alpha=style_dict['alpha1'],
        label='pre-2010 (LW)', zorder=1
    )
    f2 = ax.fill_between(
        time, m2, m3,
        color=style_dict['color2'], alpha=style_dict['alpha2'],
        label='pre-2010 (air)', zorder=1
    )
    f3 = ax.fill_between(
        time, m3, m4,
        color=style_dict['color3'], alpha=style_dict['alpha3'],
        label='post-2010 (LW)', zorder=1
    )
    f4 = ax.fill_between(
        time, m4, m5,
        color=style_dict['color4'], alpha=style_dict['alpha4'],
        label='post-2010 (air)', zorder=1
    )

    # add boundaries to distinguish time periods
    for m in [m1, m3, m5]: #[m1, m2, m3, m4, m5]:
        ax.plot(time, m, color='0.2', linewidth=0.8, zorder=2)

    # Set up axis limits
    ax.set_xlim(xlim)

    ax.grid(False)

    return ax, f0, f1, f2, f3, f4

# Style dictionary for colors and alpha values
style_dict = {'color0': '0.4', 
    'color1': (159/255, 173/255, 139/255), 'color2': (159/255, 173/255, 139/255),
    'color3': (86/255, 118/255, 153/255),  'color4': (86/255, 118/255, 153/255),
    'alpha0': 1.0, 'alpha1': 0.9, 'alpha2': 0.6, 'alpha3': 0.9, 'alpha4': 0.6,
}

fig = plt.figure(figsize=(6.5, 5))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1])

# initialize subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])
ax5 = fig.add_subplot(gs[4])
ax6 = fig.add_subplot(gs[5])

# Make plots for each scenario/reservoir combination
# We only need one set of fill objects for the legend, so capture them from the first subplot call
ax1, f0, f1, f2, f3, f4 = make_plot(ax1, scenario='SSP1-26', reservoir=['atm'],  style_dict=style_dict)
make_plot(ax2, scenario='SSP5-85', reservoir=['atm'],  style_dict=style_dict)
make_plot(ax3, scenario='SSP1-26', reservoir=['ocs','oci'], style_dict=style_dict)
make_plot(ax4, scenario='SSP5-85', reservoir=['ocs','oci'], style_dict=style_dict)
make_plot(ax5, scenario='SSP1-26', reservoir=['ocd'], style_dict=style_dict)
make_plot(ax6, scenario='SSP5-85', reservoir=['ocd'], style_dict=style_dict)

# Titles
ax1.set_title('SSP1-2.6', fontsize=11, fontweight='light', color='0.1')
ax2.set_title('SSP5-8.5', fontsize=11, fontweight='light', color='0.1')

# Set up y-axis limits and ticks
for ax in [ax1, ax2]:
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 3], minor=True)

for ax in [ax3, ax4]:
    ax.set_ylim(0, 150)
    ax.set_yticks([25, 75, 125], minor=True)

for ax in [ax5, ax6]:
    ax.set_ylim(0, 410)
    ax.set_yticks([100, 200, 300, 400], minor=False)
    ax.set_yticks([50, 150, 250, 350], minor=True)

# Label subplots (a)–(f)
axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
labels = ['a', 'b', 'c', 'd', 'e', 'f']
reservoirs = ['atmosphere','atmosphere','upper ocean','upper ocean','deep ocean','deep ocean']

for ax, label, res in zip(axes_list, labels, reservoirs):
    ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.text(0.98, 0.95, res, transform=ax.transAxes, fontsize=10, va='top', ha='right', color='0.4', fontweight='light')
    for pos in ['left', 'bottom', 'right', 'top']:
        ax.spines[pos].set_color('0.1')
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, colors='0.1', which='both')
    ax.yaxis.label.set_color('0.1')
    ax.xaxis.label.set_color('0.1')
    ax.tick_params(axis='both', which='major', labelsize=9, color='0.1')

# Create a common legend below all subplots
empty_handle = mpatches.Patch(alpha=0) #  empty handle for spacing
handles = [f0, empty_handle, f2, f1, f4, f3] # from the first subplot
labels  = ['Natural', '', 'pre-2010 (air)', 'pre-2010 (LW)',  'post-2010 (air)', 'post-2010 (LW)']  # manually defined

# Place the legend at the bottom center (adjust bbox_to_anchor for spacing)
leg = fig.legend(handles, labels, ncol=3, fontsize=9,
    loc='lower center', bbox_to_anchor=(0.5, -0.1),
    frameon=True, fancybox=False, edgecolor='0.8', labelcolor='0.1', facecolor='0.97',
)
leg.get_frame().set_linewidth(0.3)

for ax in [ax5, ax6]:
    ax.set_xlabel('Year', fontsize=9, color='0.1', labelpad=5)
fig.supylabel('Mass (Gg)', fontsize=9, color='0.1')

plt.savefig('../figures/Fig_S7.pdf', bbox_inches='tight')
