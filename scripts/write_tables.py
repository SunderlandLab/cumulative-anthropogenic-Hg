import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sigfigs(x, n=5):
    ''' Description: rounds a number to a given number of significant figures '''
    if x == 0:
        return 0
    return np.round(x, n-int(np.floor(np.log10(abs(x))))-1)

def save_df_to_pdf(df, numeric_cols=['Rate'], apply_formatting=True, row_header_height_scale:float=1.0, filename="../output_table.pdf", figsize: tuple = (7, 10)):
    """Save DataFrame to a simple PDF table with scientific notation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Remove default subplot margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    if apply_formatting:
        df[numeric_cols] = df[numeric_cols].applymap(sigfigs)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(df.columns))])

    if row_header_height_scale != 1.0:
        # Adjust first row height
        num_cols = len(df.columns)
        for col in range(num_cols):
            cell = table[0, col]
            cell.set_height(cell.get_height() * row_header_height_scale)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

# --- Table S1 - rate table ---
df = pd.read_csv('../output/main/rate_table.csv')
df.loc[(df['Compartment From']=='Terrestrial Fast') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Terrestrial Slow') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Terrestrial Protected') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Waste Fast') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Waste Slow') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Waste Protected') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Margin Sediment Burial'
df.loc[(df['Compartment From']=='Ocean Deep') & (df['Compartment To'].isnull()), 'Compartment To'] = 'Deep-Sea Sediment Burial'
save_df_to_pdf(df, numeric_cols=['Rate'], apply_formatting=True, filename='../figures/table_S1.pdf', figsize=(5, (len(df)+1)*0.3))

# -- Table S5 -- summary of ocean Hg measurements
df = pd.read_csv('../profiles/output/seawater_HgT_observation_compilation.csv')
df.loc[(df['cruiseID'] == 'GEOTRACES_Intercalibration') & (df['basin'] == 'Atlantic'), 'cruiseID'] = 'GEOTRACES Intercalibration (BATS)'
df.loc[(df['cruiseID'] == 'GEOTRACES_Intercalibration') & (df['basin'] == 'Pacific'), 'cruiseID'] = 'GEOTRACES Intercalibration (SAFe)'
df['basin'] = df['basin'].replace('Land', 'Other')

cruise_order = [
    'Kirk_2008','GN02','GN03','GN04','GN01','GN05', # Arctic
    'GEOTRACES Intercalibration (BATS)', 'GA03', 'A16N-2013', 'GA01', # Atlantic
    'IO5-2009', # Indian
    'Cossa_2009', 'GApr09', # Mediterranean
    'IOC', 'P16N', 'GEOTRACES Intercalibration (SAFe)', 'KM1128', 'GP12', 'GP16', 'SHIPPO',  # Pacific
    'GIPY06', 'P16S-2011', 'JC068', # Southern
    'GA04N', # Other 
    ]

dois = {
    'Kirk_2008': 'https://doi.org/10.1021/es801635m',
    'GN02':'https://doi.org/10.1038/s41598-018-32760-0',
    'GN03':'https://doi.org/10.1038/s41598-018-32760-0',
    'GN04':'https://doi.org/10.1016/j.marchem.2020.103855',
    'GN01':'https://doi.org/10.1016/j.marchem.2019.103686',
    'GN05':'https://doi.org/10.1016/j.marchem.2020.103855',
    'GEOTRACES Intercalibration (BATS)':'https://doi.org/10.4319/lom.2012.10.90', 
    'GA03':'https://doi.org/10.1016/j.dsr2.2014.07.004', 
    'A16N-2013':'this work', 
    'GA01':'https://doi.org/10.5194/bg-15-2309-2018',
    'IO5-2009':'this work',
    'Cossa_2009':'https://doi.org/10.4319/lo.2009.54.3.0837', 
    'GApr09':'https://doi.org/10.5285/ff46f034-f47c-05f9-e053-6c86abc0dc7e',
    'P16N':'https://doi.org/10.1029/2008GB003425', 
    'GEOTRACES Intercalibration (SAFe)':'https://doi.org/10.4319/lom.2012.10.90', 
    'KM1128':'https://doi.org/10.1002/2015GB005120', 
    'GP12':'https://doi.org/10.5285/ff46f034-f47c-05f9-e053-6c86abc0dc7e', 
    'GP16':'https://doi.org/10.1016/j.marchem.2016.09.005', 
    'SHIPPO':'https://doi.org/10.1021/acs.est.6b04238', 
    'IOC':'https://doi.org/10.1016/j.marchem.2004.02.025',
    'GIPY06':'https://doi.org/10.1016/j.gca.2011.05.001', 
    'P16S-2011':'this work', 
    'JC068':'https://doi.org/10.1002/2015GB005275',
    'GA04N':'https://doi.org/10.1002/2017GB005700', 
}

primary_basin = {
    'Kirk_2008': 'Arctic',
    'GN02':'Arctic',
    'GN03':'Arctic',
    'GN04':'Arctic',
    'GN01':'Arctic',
    'GN05':'Arctic',
    'GEOTRACES Intercalibration (BATS)':'Atlantic', 
    'GA03':'Atlantic', 
    'A16N-2013':'Atlantic', 
    'GA01':'Atlantic',
    'IO5-2009':'Indian',
    'Cossa_2009':'Mediterranean', 
    'GApr09':'Mediterranean',
    'P16N':'Pacific', 
    'GEOTRACES Intercalibration (SAFe)':'Pacific', 
    'KM1128':'Pacific', 
    'GP12':'Pacific', 
    'GP16':'Pacific', 
    'SHIPPO':'Pacific', 
    'IOC':'Pacific',
    'GIPY06':'Southern', 
    'P16S-2011':'Southern', 
    'JC068':'Southern',
    'GA04N':'Other (Black Sea)', 
}

# assign DOI based on ['cruiseID']
df['DOI'] = df['cruiseID'].map(dois)
df['primary_basin'] = df['cruiseID'].map(primary_basin)

# assign ordinal grouping based on ['cruiseID']
df['cruiseID'] = df['cruiseID'].astype('category')
df['cruiseID'] = df['cruiseID'].cat.reorder_categories(cruise_order, ordered=True)

df = df.groupby(by=['cruiseID'], as_index=False).agg({'quantity': ['unique','count'], 'year':'unique', 'basin':'unique', 'primary_basin':'first', 'DOI':'first'}).rename(columns={'quantity':'count'}).sort_values(by=['cruiseID']).reset_index(drop=True)
df.columns = df.columns.droplevel(1) # remove second level of column index
col_names = df.columns.to_list()
col_names[1] = 'Species'
col_names[2] = 'n'
df.columns = col_names

# update values for output
df['Species'] = df['Species'].apply(lambda x: ', '.join(map(str, x)).replace('[', '').replace(']', '').replace('Hg_T_D_fish', '').replace(',', '').replace(' ','').replace('Hg_T','HgT').replace('_D', ' (filtered)'))
df['year'] = df['year'].apply(lambda x: ', '.join(map(str, x)).replace('[', '').replace(']', ''))
df = df.rename(columns={'cruiseID':'Cruise ID', 'year':'Year', 'primary_basin':'Basin', 'DOI':'DOI'})
df = df[['Basin', 'Cruise ID', 'Species', 'Year', 'n', 'DOI']]

# remove 'https://doi.org/' from DOI
df['DOI'] = df['DOI'].str.replace('https://doi.org/', '')

save_df_to_pdf(df, numeric_cols=[], apply_formatting=True, row_header_height_scale=1.5, filename='../figures/table_S5.pdf', figsize=(5, (len(df)+1)*0.3))

# -- Table S6 -- seawater Hg estimates by basin
df = pd.read_csv('../profiles/output/budget_table_all_depths.csv')
df.rename(columns={'basin':'Basin', 'volume [L]': 'Volume [L]', 
    'conc_p50 [pM]': 'Concentration (pmol L$^{-1}$)\n median [IQR]',
    'mass_p50 [Gg]': 'Mass (Gg)\n median [IQR]', 'count': r'$n$'}, inplace=True)
df = df.drop(columns=['Depths'])
save_df_to_pdf(df, numeric_cols=['Volume [L]'], apply_formatting=True, row_header_height_scale=2.2, filename='../figures/table_S6.pdf', figsize=(5, (len(df)+1)*0.3))

# -- Table S7 -- volume-weighted seawater Hg estimate quantiles for upper and deep ocean
df = pd.read_csv('../profiles/output/seawater_concentration_quantiles.csv')
save_df_to_pdf(df, numeric_cols=['p5','p25','p50','p75','p95'], apply_formatting=True, row_header_height_scale=1.5, filename='../figures/table_S7.pdf', figsize=(5, (len(df)+1)*0.3))

# -- Table S8 -- future fluxes in 2100
tmp = pd.read_csv('../output/main/budget_table_2100_agg1_sector_SSP1-26_1510_2300.csv')
df = tmp[['from', 'to', 'flux [Mg/yr]']].copy()
df.rename(columns={'from': 'Compartment From', 
'to': 'Compartment To',
'flux [Mg/yr]': 'Flux (Mg a$^{-1}$)\nSSP1-2.6'}, inplace=True)
tmp = pd.read_csv('../output/main/budget_table_2100_agg1_sector_SSP5-85_1510_2300.csv')
df['Flux (Mg a$^{-1}$)\nSSP5-8.5'] = tmp['flux [Mg/yr]'].copy()
# assign sig figs to fluxes
df['Flux (Mg a$^{-1}$)\nSSP1-2.6'] = df['Flux (Mg a$^{-1}$)\nSSP1-2.6'].apply(lambda x: sigfigs(x, n=2))
df['Flux (Mg a$^{-1}$)\nSSP5-8.5'] = df['Flux (Mg a$^{-1}$)\nSSP5-8.5'].apply(lambda x: sigfigs(x, n=2))
# if value is >1, format as int
df['Flux (Mg a$^{-1}$)\nSSP1-2.6'] = df['Flux (Mg a$^{-1}$)\nSSP1-2.6'].apply(lambda x: str(int(x)) if x > 10 else x)
df['Flux (Mg a$^{-1}$)\nSSP5-8.5'] = df['Flux (Mg a$^{-1}$)\nSSP5-8.5'].apply(lambda x: str(int(x)) if x > 10 else x)

save_df_to_pdf(df, filename='../figures/table_S8.pdf', numeric_cols=[], figsize=(7, 10), row_header_height_scale=1.5)