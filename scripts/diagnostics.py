import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boxey as bx
from tenbox_boxey import init_model
import json

def sigfigs(x, n=5):
    ''' Description: rounds a number to a given number of significant figures '''
    if x == 0:
        return 0
    return np.round(x, n-int(np.floor(np.log10(abs(x))))-1)

def get_magnitude(x):
    ''' Description: returns the order of magnitude of a real positive number '''
    if type(x) == str:
        return x
    if x == 0:
        return 0
    assert x >= 0, 'x must be positive'
    return int(np.floor(np.log10(abs(x))))

def assign_significant_digits(x):
    ''' Description: assigns significant digits to a number '''
    if type(x) == str:
        return x
    magnitude = get_magnitude(x)
    if magnitude < 0:
        return sigfigs(x, n=3)
    elif magnitude <= 1:
        return int(sigfigs(x, n=2))

    return int(sigfigs(x, n=2))

# function to get emission types for a given set of reservoirs -- used by `get_legacy()` below
def get_emissions_to_atm(model, solution: pd.DataFrame):
    
    flux_collections = {'waste_volatilization' : ['k_We_wf', 'k_We_ws', 'k_We_wa'],
                        'ocean_evasion'        : ['k_Oc_ev'],
                        'ocean_Hg0_uptake'     : ['k_A_oHg0'],
                        'terrestrial_emissions': ['k_Te_p','k_Te_rf','k_Te_bbf','k_Te_rs','k_Te_bbs','k_Te_ra','k_Te_bba'],
                        }

    output = {'Year': solution['Year']}
    
    for key in flux_collections.keys():
        for process in flux_collections[key]:
            # optional -- check that emission is to atmosphere
            # assert model.processes[process].compartment_to == 'atm'
            # get rate coefficient
            k = model.processes[process].get_k()
            # source compartment
            compartment_from = model.processes[process].compartment_from
            M = solution[compartment_from]
            E = k*M

            if key not in output.keys():
                output[key] = E
            else:
                output[key] += E
            
    return pd.DataFrame(output)

def make_diagnostic_table(model, solution, natural_emission_vector=[230.,0,0,0,0,0,50.,0,0,0]):

    # list of compartments to get diagnostics for
    diagnostics_compartments = ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd']

    # define years of "present" and "pre-industrial"
    present_year = 2010
    preind_year  = 1850

    # get steady for input natural emission vector
    steady = model.get_steady_state(input_vector=natural_emission_vector)

    present_reservoirs = solution.get_reservoirs(time=present_year, names=diagnostics_compartments)
    preind_reservoirs  = solution.get_reservoirs(time=preind_year,  names=diagnostics_compartments)

    # create empty dataframe to store output reservoir masses and enrichment factors
    df_out = pd.DataFrame(columns=['Reservoir Name','Mass Natural (Gg)', 'Mass Pre-Industrial (Gg)', 'Mass Present (Gg)', 'EF vs. All-Time', 'EF vs. Pre-industrial'])

    for i, name in enumerate(diagnostics_compartments):
        nat   = steady[i]/1000
        pres  = present_reservoirs[i]/1000
        preind = preind_reservoirs[i]/1000
        df_tmp = pd.DataFrame({'Reservoir Name':          [name],
                               'Mass Natural (Gg)':       [np.round(nat,3)], 
                               'Mass Pre-Industrial (Gg)':[np.round(preind,3)],
                               'Mass Present (Gg)':       [np.round(pres,3)], 
                               'EF vs. All-Time':         [np.round(pres/nat, 3)], 
                               'EF vs. Pre-industrial':   [np.round(pres/preind,3)]})
        df_out = pd.concat((df_out, df_tmp))

    # use `sigfigs` function to round values
    for col in df_out.columns[1:]: # skip 'Reservoir Name'
        df_out[col] = df_out[col].apply(sigfigs)

    return df_out.reset_index(drop=True)

def make_constraint_table(model, solution, natural_emission_vector=[230.,0,0,0,0,0,50.,0,0,0], oci_depth_km=1.5, display_output=True):

    # list of compartments to get diagnostics for
    diagnostics_compartments = ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd']

    solution_df = solution.to_dataframe()

    # define years of "present" and "pre-industrial"
    present_year  = 2010
    preind_year   = 1850
    preanth_year = -2000

    present_reservoirs = solution_df[solution_df['Year']==present_year]
    preind_reservoirs  = solution_df[(solution_df['Year']>=1800) & (solution_df['Year']<=1860)].mean()
    extended_20thCmax_reservoirs = solution_df[(solution_df['Year']>=1950) & (solution_df['Year']<=1990)].mean()
    preanth_reservoirs = solution_df[solution_df['Year']==preanth_year].mean()

    # ---------------------------
    # compare to solid constriants
    # ---------------------------    
    # Setup structure of constraints table
    constraints = pd.DataFrame(columns=['Constraint Name', 'Constraint Min', 'Constraint Max', 'Constraint Central', 'Model', 'Notes'])

    # -- Add present atmospheric Hg burden constraint --
    df_tmp = pd.DataFrame({'Constraint Name':    ['Atmospheric Burden (Gg)'], 
                           'Constraint Min':     [4.0], 
                           'Constraint Max':     [4.5],
                           'Constraint Central': [4.3], 
                           'Model':              [present_reservoirs['atm'].item()*1e-3], 'Notes':''})
    constraints = pd.concat((constraints, df_tmp))

    # -- Add preindustrial to 20thC max atmospheric enrichment factor constraint --
    #peak_20C = solution.to_dataframe()
    peak_20C = solution_df[(solution_df['Year']>=1900) & (solution_df['Year']<2000)] #peak_20C[(peak_20C['Year']<2000) & (peak_20C['Year']>=1900) ]
    idx_max  = np.argmax(peak_20C['atm'])
    peak_yr_20C, peak_val_20C  = peak_20C.iloc[idx_max]['Year'], peak_20C.iloc[idx_max]['atm']
    #peak_EF_20C  = peak_val_20C/preind_reservoirs[0]
    # calculate EF_preind as ratio between extended 20th C max (1950 - 1990) and broad preindustrial period (1800-1860)
    EF_preind = (extended_20thCmax_reservoirs['atm']/preind_reservoirs['atm'])

    df_tmp = pd.DataFrame({'Constraint Name':    ['Preind atm. EF'], 
                           'Constraint Min':     [1.6],
                           'Constraint Max':     [3.72],
                           'Constraint Central': [2.66],
                           'Model':              [EF_preind],
                           'Notes':              ['mean +/- 1 std. dev. from Li et al. (2022)'],
                           })
    constraints = pd.concat((constraints, df_tmp))

    # -- Add preanthropogenic to 20thC max atmospheric enrichment factor constraint --
    alltime_EF = (extended_20thCmax_reservoirs['atm']/preanth_reservoirs['atm'])
    df_tmp = pd.DataFrame({'Constraint Name':     ['Alltime atm. EF'],
                            'Constraint Min':     [2.15],
                            'Constraint Max':     [14.43],
                            'Constraint Central': [8.29],
                            'Model':              [alltime_EF],
                            'Notes':              ['mean +/- 1 std. dev. from Li et al. (2022)'],
                            })
    constraints = pd.concat((constraints, df_tmp))

    # -- Add present ocean concentrations constraint --
    # -- note: future applications should consider using volume fractions calculated directly
    #          from ETOPO bathymetry data, rather than using assumption that surfintvol can be
    #          calculated from ocean surface area
    km3_to_L           = 1e12    # km3 -> L
    kg_to_mol          = 1e3/200.59 # kg Hg -> mol Hg

    # instead get ocean volume from GEBCO bathymetry used to construct measurement-based inventories
    Rocsvol_L  = 2.0e19 # L (surface ocean volume; 0-55m depth; GEBCO bathymetry)
    surfintvol = 4.9e20 # L (surface + intermediate ocean volume; 0-1500m depth; GEBCO bathymetry)
    deepvol    = 8.3e20 # L (deep ocean volume; >1500m depth; GEBCO bathymetry)

    conc_deepoc = present_reservoirs['ocd'].item()*1e3*kg_to_mol*1e12/deepvol # Mg -> kg -> mol -> pmol -> pM
    conc_intoc  = np.sum(present_reservoirs[['ocs','oci']].values)*1e3*kg_to_mol*1e12/surfintvol

    # [Min, Max] bounds from Sunderland and Mason (2007). Central estimate slightly lower than
    # Sunderland and Mason values (0-1500m: 1.2 pM; >1500m: 1.4 pM) to reflect more recent data from 
    # GEOTRACES era. Lamborg et al. (2014) estimates are on lower end of Sunderland and Mason (2007) estimates. 
    df_tmp = pd.DataFrame({'Constraint Name':    ['Upper Ocean Conc. (pM)', 'Deep Ocean Conc. (pM)'], 
                           'Constraint Min':     [0.49, 0.64],
                           'Constraint Max':     [1.65, 1.66],
                           'Constraint Central': [1.0, 1.1],
                           'Model':              [conc_intoc, conc_deepoc],
                           'Notes':              [f'Upper ocean depth: {oci_depth_km} km','']})
    constraints = pd.concat((constraints, df_tmp))

    # use `sigfigs` function to round values
    constraints['Model'] = constraints['Model'].apply(sigfigs, n=5)

    if display_output:
        print(constraints)
    
    return constraints.reset_index(drop=True)

def make_budget_construction_table(model_parameters, add_rate_column=False):
    ''' 
    Description: 
    ------------
        Creates a table of budget construction for the 10-box model
    
    Arguments:
    ----------
        model_parameters (dict): dictionary of model parameters
        add_rate_column (bool): if True, add a column for rate coefficients
    
    Returns:
    --------
        df: DataFrame containing the budget construction table in format of Table S2 from Amos et al. (2014)
    '''

    # Make table of present-day reservoirs and fluxes used to 
    # calculate first-order rate coefficients
    # in 10-box model of Hg global biogeochemical cycling.
    # This is based off of Table S2 from Amos et al. (2014)
    # https://pubs.acs.org/doi/suppl/10.1021/es502134t/suppl_file/es502134t_si_001.pdf

    # read display descriptions from json
    with open('../inputs/meta/budget_display_descriptions.json', 'r') as f:
        budget_display_descriptions = json.load(f)

    # Step through values that should be in model_parameters. If value not in model_parameters,
    # then assign value which was used in tenbox_boxey.py. As needed, calculate values that depend 
    # on existing values. Note that this is fragile, since it requires that defaults mention those
    # assigned in tenbox_boxey.py.
    model_parameters['fsoil']  = model_parameters.get('fsoil', (1. - model_parameters['fveg']) )
    model_parameters['fCfast'] = model_parameters.get('fCfast', 0.2185)
    model_parameters['fCslow'] = model_parameters.get('fCslow', 0.5057)
    model_parameters['fCarmored'] = model_parameters.get('fCarmored', (1. - (model_parameters['fCfast']+model_parameters['fCslow'])) )

    model_parameters['IHgD_pristine'] = model_parameters.get('IHgD_pristine', 78)
    model_parameters['IHgP_pristine'] = model_parameters.get('IHgP_pristine', 659)
    model_parameters['Te_riv_margin'] = model_parameters.get('Te_riv_margin', (model_parameters['IHgD_pristine'] + model_parameters['IHgP_pristine']) )
    model_parameters['f_HgPexport']   = model_parameters.get('f_HgPexport', 0.07)

    # partition total river flux among fast, slow, protected if not specified
    model_parameters['T_riv_f'] = model_parameters.get('T_riv_f', (model_parameters['Te_riv_margin'] * (model_parameters['fveg'] + (model_parameters['fsoil']*model_parameters['fCfast']))))
    model_parameters['T_riv_s'] = model_parameters.get('T_riv_s', (model_parameters['Te_riv_margin'] * (model_parameters['fsoil']*model_parameters['fCslow'])))
    model_parameters['T_riv_a'] = model_parameters.get('T_riv_a', (model_parameters['Te_riv_margin'] * (model_parameters['fsoil']*model_parameters['fCarmored'])))

    model_parameters['Te_riv_ocean'] = model_parameters.get('Te_riv_ocean', (model_parameters['IHgD_pristine'] + model_parameters['f_HgPexport']*model_parameters['IHgP_pristine']) )
    model_parameters['O_riv_f'] = model_parameters.get('O_riv_f', (model_parameters['Te_riv_ocean'] * (model_parameters['fveg'] + (model_parameters['fsoil']*model_parameters['fCfast']))))
    model_parameters['O_riv_s'] = model_parameters.get('O_riv_s', (model_parameters['Te_riv_ocean'] * (model_parameters['fsoil']*model_parameters['fCslow'])))
    model_parameters['O_riv_a'] = model_parameters.get('O_riv_a', (model_parameters['Te_riv_ocean'] * (model_parameters['fsoil']*model_parameters['fCarmored'])))

    # get river flux to margin sediment
    model_parameters['Sed_riv_f'] = model_parameters['T_riv_f'] - model_parameters['O_riv_f']
    model_parameters['Sed_riv_s'] = model_parameters['T_riv_s'] - model_parameters['O_riv_s']
    model_parameters['Sed_riv_a'] = model_parameters['T_riv_a'] - model_parameters['O_riv_a']

    # partition total biomass burning flux among fast, slow, protected if not specified
    model_parameters['Te_bbf'] = model_parameters.get('Te_bbf', (model_parameters['tot_bb'] * (model_parameters['fveg'] + (model_parameters['fsoil']*model_parameters['fCfast']))) )
    model_parameters['Te_bbs'] = model_parameters.get('Te_bbs', (model_parameters['tot_bb'] * (model_parameters['fsoil']*model_parameters['fCslow'])) )
    model_parameters['Te_bba'] = model_parameters.get('Te_bba', (model_parameters['tot_bb'] * (model_parameters['fsoil']*model_parameters['fCarmored'])) )

    # -- order reservoirs and deposition terms according to Table S2 from Amos et al. (2014)
    value_order = ['Ratm', 'Dep_tHgII', 'Dep_tHg0', 'Dep_oHgII', 'Upt_oHg0',
                   'Rocs', 'Ev_Hg0_ocs', 'ps_ocs', 'vert_ocsi',
                   'Roci', 'ps_oci', 'vert_ocis', 'vert_ocid',
                   'Rocd', 'ps_ocd', 'vert_ocdi',
                   'Rtf', 'Te_rf', 'Te_p', 'Te_bbf', 'Te_exfs', 'Te_exfa', 'T_riv_f',
                   'Rts', 'Te_rs', 'Te_bbs', 'Te_exsf', 'Te_exsa', 'T_riv_s',
                   'Rta', 'Te_ra', 'Te_bba', 'Te_exaf', 'T_riv_a',]

    # create pandas dataframe with model parameters
    df = pd.DataFrame(columns=['Parameter Name', 'Description', 'Value'])
    for v in value_order:
        df_tmp = pd.DataFrame({'Parameter Name': [v], 'Description': [budget_display_descriptions[v]], 'Value': [model_parameters[v]]})
        df = pd.concat((df, df_tmp))

    df = pd.concat((df, pd.DataFrame({'Parameter Name': ['-- other terms --'], 'Description': ['--'], 'Value': ['--']})))

    # add model_parameter keys not in value_order
    missing_keys = [x for x in model_parameters.keys() if x not in value_order]
    for v in missing_keys:
        df_tmp = pd.DataFrame({'Parameter Name': [v], 'Description': [budget_display_descriptions[v]], 'Value': [model_parameters[v]]})
        df = pd.concat((df, df_tmp))

    df = df.reset_index(drop=True)

    # -- add column containing rate coefficients
    if add_rate_column:
        # define source reservoirs for each process
        source_reservoirs = {
            'Dep_tHgII':'Ratm', 'Dep_tHg0':'Ratm', 'Dep_oHgII':'Ratm', 'Upt_oHg0':'Ratm',
            'Ev_Hg0_ocs':'Rocs', 'ps_ocs':'Rocs', 'vert_ocsi':'Rocs',
            'ps_oci':'Roci', 'vert_ocis':'Roci', 'vert_ocid':'Roci',
            'ps_ocd':'Rocd', 'vert_ocdi':'Rocd',
            'Te_rf':'Rtf', 'Te_p':'Rtf', 'Te_bbf':'Rtf', 'Te_exfs':'Rtf', 'Te_exfa':'Rtf', 'T_riv_f':'Rtf',
            'Te_rs':'Rts', 'Te_bbs':'Rts', 'Te_exsf':'Rts', 'Te_exsa':'Rts', 'T_riv_s':'Rts',
            'Te_ra':'Rta', 'Te_bba':'Rta', 'Te_exaf':'Rta', 'T_riv_a':'Rta',
            # -- other terms --
            'O_riv_f':'Rtf', 'O_riv_s':'Rts', 'O_riv_a':'Rta',
            'Sed_riv_f':'Rtf', 'Sed_riv_s':'Rts', 'Sed_riv_a':'Rta',
        }        
        # Create a new column 'source_reservoir_mass' by mapping 'Parameter Name' through 'source_reservoirs'
        df['source_reservoir_mass'] = df['Parameter Name'].map(source_reservoirs)
        # Calculate rates
        df['Rate'] = np.where(df['source_reservoir_mass'].notna(), df['Value'] / df['source_reservoir_mass'].map(model_parameters), np.nan)

    # assign significant digits, but only if value is numeric
    df['Value'] = df['Value'].apply(assign_significant_digits)

    if add_rate_column:
        # Format 'Rate' in scientific notation with 1 decimal place
        df['Rate'] = df['Rate'].apply(lambda x: f'{x:.1e}' if pd.notnull(x) else np.nan)   

        return df[['Description', 'Value', 'Rate']]
    else:
        return df[['Description', 'Value']]

def make_budget_table_from_model(model, solution, year=2010, aggregation_level=0):

      '''
      Description: 
      ------------
      This function creates a budget table from a boxey `model` and `solution` object pair. The budget table is a DataFrame that
      contains the mass fluxes between compartments for a given year.

      Parameters:
      -----------
      model : boxey.Model
            A boxey model object
      solution : boxey.Solution
            A boxey solution object
      year : int
            The year for which the budget table is created
      aggregation_level : int
            The level of aggregation for the budget table. The budget table can be aggregated at four levels:
                  0. no aggregation
                  1. compartments are grouped by source reservoir [10-compartment model]
                  2. compartments are grouped by supergrouping ['Atmosphere', 'Terrestrial', 'Ocean', 'Waste', 'Sediment Burial']
                  3. compartments are grouped by supergrouping, and internal fluxes are not displayed

      Returns:
      --------
      df : pd.DataFrame
            A DataFrame containing the budget table for the given year and aggregation level
      '''
      
      assert aggregation_level in [0,1,2,3], 'aggregation_level must be 0, 1, 2, or 3'

      process_descriptions = json.load(open(f'../inputs/meta/boxey_process_descriptions.json'))

      # -- get entries to budget table
      process_names       = [k for k in solution.get_fluxes(year).keys()]
      fluxes              = [k for k in solution.get_fluxes(year).values()]
      compartment_froms   = [model.processes[k].compartment_from for k in process_names]
      compartment_tos     = [model.processes[k].compartment_to for k in process_names]
      descriptions        = [process_descriptions[k] for k in process_names]
      source_mass         = [solution.get_reservoir(time=year, name=k) for k in compartment_froms]

      df = pd.DataFrame({'name':process_names, 'from':compartment_froms, 'to': compartment_tos, 
            'source mass [Mg]':source_mass, 'flux [Mg/yr]':fluxes, #'rate [1/yr]': np.array(fluxes)/np.array(source_mass),
            'year':year, 'description':descriptions, })
            
      df = df.fillna('') # replace None with empty string

      # sort by compartment
      df['from'] = pd.Categorical(df['from'], ['atm', 'tf','ts','ta', 'ocs','oci','ocd', 'wf','ws','wa', '',])
      df['to']   = pd.Categorical(df['to'],   ['atm', 'tf','ts','ta', 'ocs','oci','ocd', 'wf','ws','wa', '',])
      df = df.sort_values(['from', 'to'])

      # rename compartments
      comp_names = {'atm': 'Atmosphere', 
            'tf':'Terrestrial Fast', 'ts':'Terrestrial Slow', 'ta':'Terrestrial Protected',
            'ocs':'Ocean Surface', 'oci':'Ocean Intermediate', 'ocd':'Ocean Deep',
            'wf':'Waste Fast', 'ws':'Waste Slow', 'wa':'Waste Protected', '' :'Sediment Burial',}

      df['from'] = df['from'].map(comp_names)
      df['to']   = df['to'].map(comp_names)

      # ------------------------------------------------------------------------------------------------
      # -- condense table by applying some grouping over 'from' and 'to'
      # ------------------------------------------------------------------------------------------------
      if aggregation_level == 1:
        df = df.groupby(by=['from', 'to', 'year', 'source mass [Mg]']).sum(numeric_only=True).reset_index()

      # -- aggregate over supergrouping in "more condensed" and "most condensed" formatting
      elif (aggregation_level == 2) or (aggregation_level == 3):
            # here, supergroups are ['Atmosphere', 'Terrestrial', 'Ocean', 'Waste', 'Sediment Burial']
            supergrouping = {
                  'Atmosphere':'Atmosphere',
                  'Terrestrial Fast':'Terrestrial', 'Terrestrial Slow':'Terrestrial', 'Terrestrial Protected':'Terrestrial',
                  'Ocean Surface':'Ocean', 'Ocean Intermediate':'Ocean', 'Ocean Deep':'Ocean',
                  'Waste Fast':'Waste', 'Waste Slow':'Waste', 'Waste Protected':'Waste',
                  'Sediment Burial':'Sediment Burial',}

            # get sum of unique source masses by supergroup
            supergroup_masses = df.groupby(by=['from', 'year']).agg({'source mass [Mg]':'first'}).reset_index()
            supergroup_masses = supergroup_masses.groupby(by='from').sum(numeric_only=True).reset_index()
            supergroup_masses['from'] = supergroup_masses['from'].map(supergrouping)
            supergroup_masses = supergroup_masses.groupby(by=['from','year']).sum(numeric_only=True).reset_index()

            df['from'] = df['from'].map(supergrouping)
            df['to']   = df['to'].map(supergrouping)

            # sort supergroups
            df['from'] = pd.Categorical(df['from'], ['Atmosphere', 'Terrestrial', 'Ocean', 'Waste', 'Sediment Burial'])
            df['to']   = pd.Categorical(df['to'],   ['Atmosphere', 'Terrestrial', 'Ocean', 'Waste', 'Sediment Burial'])
            df = df.sort_values(['from', 'to'])

            df = df.groupby(by=['from', 'to', 'year']).sum(numeric_only=True).reset_index()[['from','to','year','flux [Mg/yr]']]
            df = pd.merge(df, supergroup_masses, on=['from', 'year'], how='left')

            # -- do not display internal fluxes in most condensed formatting
            if aggregation_level == 3:
                  df = df[df['from'] != df['to']]

      # remove rows with zero flux
      df = df[df['flux [Mg/yr]'] != 0]
      # ------------------------------------------------------------------------------------------------

      return df

def make_rate_table(model):
    ''' Description: creates a table of rate coefficients for the 10-box model '''

    comp_names = {'atm': 'Atmosphere',
                  'tf':  'Terrestrial Fast',
                  'ts':  'Terrestrial Slow',
                  'ta':  'Terrestrial Protected',
                  'ocs': 'Ocean Surface',
                  'oci': 'Ocean Intermediate',
                  'ocd': 'Ocean Deep',
                  'wf':  'Waste Fast',
                  'ws':  'Waste Slow',
                  'wa':  'Waste Protected',
                  ''  : 'Sediment Burial',
                  }

    rate_dict = {'Compartment From':[], 'Compartment To':[], 'Rate':[]}
    for process in model.processes:
        k         = model.processes[process].get_k(process)
        comp_from = model.processes[process].compartment_from
        comp_to   = model.processes[process].compartment_to
        if comp_to == None:
            comp_to = ''
        
        rate_dict['Compartment From'].append(comp_from)
        rate_dict['Compartment To'].append(comp_to)
        rate_dict['Rate'].append(k)

    rate_df = pd.DataFrame(rate_dict)
    rate_df = rate_df.groupby(['Compartment From', 'Compartment To']).sum().reset_index()
    # assign ordering to Compartment From and Compartment To
    rate_df['Compartment From'] = pd.Categorical(rate_df['Compartment From'], ['atm', 'tf','ts','ta', 'ocs','oci','ocd', 'wf','ws','wa', 'Sediment Burial'])
    rate_df['Compartment To']   = pd.Categorical(rate_df['Compartment To'],   ['atm', 'tf','ts','ta', 'ocs','oci','ocd', 'wf','ws','wa', 'Sediment Burial'])

    rate_df = rate_df.sort_values(['Compartment From', 'Compartment To'])

    # rename with compartment names
    rate_df['Compartment From'] = rate_df['Compartment From'].map(comp_names)
    rate_df['Compartment To']   = rate_df['Compartment To'].map(comp_names)
    return rate_df

def illustrate_reservoirs(res_df_path:str, year:int=2010, n_sig:int=4):
    ''' 
    Description:
    ------------
        Illustrates the reservoirs of the 10-box model in simple figure of annotated rectangles
    
    Arguments:
    ----------
        res_df_path (str): path to the csv file containing the reservoir data
        year (int): year to illustrate the reservoirs for
        n_sig (int): number of significant figures to use in annotation

    Returns:
    --------
        fig, ax: figure and axis objects

    Example:
    --------
        fig, ax = illustrate_reservoirs(res_df_path='./output/main/all_inputs_output_sector_SSP1-26_1510_2010.csv', year=2010)
    '''

    df = pd.read_csv(res_df_path)

    row = df[df['Year']==year]

    # -- arguments for annotation
    margin = 0.02
    ha = 'left'
    va = 'top'

    patch_arg_dict = {'atm': {'x0_patch': 0, 'y0_patch': 0.75, 'width_patch': 1, 'height_patch': 0.25, 'color': 'whitesmoke', 'label': f'Atmosphere\n'+str(int(sigfigs(row['atm'].item(), n_sig))), 'textcolor': 'k'},
        'ocs': {'x0_patch': 0.66, 'y0_patch': 0.5, 'width_patch': 0.34, 'height_patch': 0.25, 'color': 'lightblue', 'label': f'Surf. Ocean\n'+str(int(sigfigs(row['ocs'].item(), n_sig))), 'textcolor': 'k'},
        'oci': {'x0_patch': 0.66, 'y0_patch': 0.25, 'width_patch': 0.34, 'height_patch': 0.25, 'color': 'blue', 'label': f'Int. Ocean\n'+str(int(sigfigs(row['oci'].item(), n_sig))), 'textcolor': 'w'},
        'ocd': {'x0_patch': 0.66, 'y0_patch': 0, 'width_patch': 0.34, 'height_patch': 0.25, 'color': 'navy', 'label': f'Deep Ocean\n'+str(int(sigfigs(row['ocd'].item(), n_sig))), 'textcolor': 'w'},
        'tf': {'x0_patch': 0.33, 'y0_patch': 0.5, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'yellowgreen', 'label': f'Fast Soil\n'+str(int(sigfigs(row['tf'].item(), n_sig))), 'textcolor': 'k'},
        'ts': {'x0_patch': 0.33, 'y0_patch': 0.25, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'green', 'label': f'Slow Soil\n'+str(int(sigfigs(row['ts'].item(), n_sig))), 'textcolor': 'w'},
        'ta': {'x0_patch': 0.33, 'y0_patch': 0, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'darkgreen', 'label': f'Prot. Soil\n'+str(int(sigfigs(row['ta'].item(), n_sig))), 'textcolor': 'w'},
        'wf': {'x0_patch': 0, 'y0_patch': 0.5, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'tan', 'label': f'Fast Waste\n'+str(int(sigfigs(row['wf'].item(), n_sig))), 'textcolor': 'k'},
        'ws': {'x0_patch': 0, 'y0_patch': 0.25, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'peru', 'label': f'Slow Waste\n'+str(int(sigfigs(row['ws'].item(), n_sig))), 'textcolor': 'w'},
        'wa': {'x0_patch': 0, 'y0_patch': 0, 'width_patch': 0.33, 'height_patch': 0.25, 'color': 'sienna', 'label': f'Prot. Waste\n'+str(int(sigfigs(row['wa'].item(), n_sig))), 'textcolor': 'w'},
        }

    for k, v in patch_arg_dict.items():
        patch_arg_dict[k]['kwargs'] = {'ha': 'left', 'va': 'top'}

    # plot series of rectangles
    fig, ax = plt.subplots(figsize=(5,2.5))
    # -- 
    for k, v in patch_arg_dict.items():
        # -- make the patch
        ax.add_patch(plt.Rectangle((v['x0_patch'], v['y0_patch']), v['width_patch'], v['height_patch'], color=v['color']))
        
        # -- annotate
        ax.text(v['x0_patch'] + margin, v['y0_patch'] + v['height_patch'] - margin, v['label'], ha='left', va='top', color=v['textcolor'])

    # -- remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax