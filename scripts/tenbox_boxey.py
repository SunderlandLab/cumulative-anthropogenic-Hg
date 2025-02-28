import numpy as np
from scipy.interpolate import interp1d
from boxey import Model, Process


def init_model(input_dict={}):

    compartments = ['atm', 'tf', 'ts', 'ta', 'ocs', 'oci', 'ocd', 'wf', 'ws', 'wa',]

    model = Model(compartments)

# --------------------------------------------------------------------------
# Reservoirs estimates for present day (Mg)
# --------------------------------------------------------------------------

# Atmosphere
    # Shah et al.(2021)
    Ratm = input_dict.get('Ratm', 4000)

# Ocean
    # Soerensen et al. (2010)
    Rocs = input_dict.get('Rocs', 2910)
    # Sunderland and Mason (2007)
    Roci = input_dict.get('Roci', 134000)  
    # Sunderland and Mason (2007)
    Rocd = input_dict.get('Rocd', 220649)

#  Terrestrial
    # Leaf, fast and intermediate pools from Smith-Downey et al (2010)
    Rtf = input_dict.get('Rtf', 9620)
    # Slow pool from Smith-Downey et al. (2010)
    Rts = input_dict.get('Rts', 34900)
    # Protected pool representing remaining Hg observed in 
    # global topsoil (0 - 30cm); Lim et al. (2020) estimate of 1086 Gg total
    Rta = input_dict.get('Rta', 1041480)

# --------------------------------------------------------------------------
# Fluxes for present day (Mg/year)
# --------------------------------------------------------------------------

# Atmosphere
    # Hg(II) deposition to the ocean
    Dep_oHgII = input_dict.get('Dep_oHgII', 3900) # 3900 Shah et al. (2021); 2350 - 3900 Sonke et al. (2023)

    # Hg(II) deposition to land
    Dep_tHgII = input_dict.get('Dep_tHgII', np.nan) # 1600 Shah et al. (2021); 1600 Sonke et al. (2023)
    # Hg(0) deposition to land
    Dep_tHg0  = input_dict.get('Dep_tHg0', np.nan) # 1200 Shah et al. (2021); 2850 Sonke et al. (2023)

# Hg0 air - sea exchange
    # net evasion from surface ocean to atmosphere (Shah et al. 2021)
    # netEv_Hg0 = input_dict.get('netEv_Hg0', 2800)
    # gross ocean evasion to the atmosphere (Shah et al. 2021)
    Ev_Hg0_ocs = input_dict.get('Ev_Hg0_ocs', np.nan) # 4800 - 8300 Sonke et al. 2023
    # gross uptake of Hg0 from the atmopshere (2000 in Shah et al. 2021)
    # 1950 - 3900 Sonke et al. 2023
    Upt_oHg0 = input_dict.get('Upt_oHg0', np.nan) #Ev_Hg0_ocs - netEv_Hg0

# Surface ocean
    ps_ocs = input_dict.get('ps_ocs', 3320) # particle settling
    # gross detrainment flux, surface to intermediate
    vert_ocsi = input_dict.get('vert_ocsi', 5100)

# Intermediate ocean
    Kd_vert = input_dict.get('Kd_vert', 1e-5) # m2/s (Talley Descriptive Oceanography eddy diffusivity S7.4)
    # 1e-5 m2/s diffusivity, ~1000m vertical scale, unit change to per year
    k_diffi = input_dict.get('k_diffi', Kd_vert*3600*24*365/(1000**2))
    ps_oci  = input_dict.get('ps_oci', 480) # particle settling
    # vertical seawater flow, intermediate to surface
    vert_ocis = input_dict.get('vert_ocis', 7100)
    # vertical seawater flow, intermediate to deep
    vert_ocid = input_dict.get('vert_ocid', 335)
    vert_ocid += k_diffi*Roci # add diffusive flux to vertical flow

# Deep ocean
    k_diffd = input_dict.get('k_diffd', 3*Kd_vert*3600*24*365/(3000**2)) # vertical diffusivity is about 3x higher deep
    # particle settling, burial in deep sediments
    ps_ocd = input_dict.get('ps_ocd', 220)
    # vertical sea water flow, deep to intermediate
    vert_ocdi = input_dict.get('vert_ocdi', 175)
    vert_ocdi += k_diffd*Rocd  # add diffusive flux to vertical flow

    # rivers and biomass burning: assuming 75# vegetation (all fast) + 25#
    # soils (fast,slow,armored) partitioned by C storage
    fveg  = input_dict.get('fveg',  0.75)            # fraction to vegetation
    fsoil = input_dict.get('fsoil', 1. - fveg)
    assert np.abs((fveg + fsoil) - 1.) < 1e-6, 'fveg + fsoil must equal 1'

    # fraction of carbon in each pool (Smith-Downey et al., 2010)
    fCfast    = input_dict.get('fCfast', 0.2185)          # fast reservoir
    fCslow    = input_dict.get('fCslow', 0.5057)          # slow reservoir
    fCarmored = input_dict.get('fCarmored', 0.2758)       # armored reservoir
    assert np.abs((fCfast + fCslow + fCarmored) - 1) < 1e-6, 'fCfast + fCslow + fCarmored must equal 1'

    # total biomass burning
    tot_bb   = input_dict.get('tot_bb', np.nan) # 450 Sonke et al., 2023; 675 Friedli et al. (2009)
    # total evasion due to respiration of organic matter
    tot_resp = input_dict.get('tot_resp', np.nan) # 1100 Sonke et al., 735 Smith-Downey et al. (2010)

# Fast terrestrial  
    # evasion due to respiration of organic carbon
    E_Te_rf  = input_dict.get('Te_rf', tot_resp*460/735)
    # biomass burning, fast pool to atmosphere
    E_Te_bbf = input_dict.get('Te_bbf', tot_bb*(fveg + (fsoil*fCfast)) )
    # exchange among soil pools, fast pool to slow pool
    Te_exfs  = input_dict.get('Te_exfs', 325)
    # exchange among soil pools, fast pool to armored pool
    Te_exfa  = input_dict.get('Te_exfa', 9)

# Slow terrestrial
    # evasion due to respiration of organic carbon
    E_Te_rs = input_dict.get('Te_rs', tot_resp*250/735)
    # biomass burning, slow pool to atmosphere
    E_Te_bbs = input_dict.get('Te_bbs', tot_bb*(fsoil*fCslow) )
    # exchange among soil pools, slow pool to fast pool
    Te_exsf = input_dict.get('Te_exsf', 205)
    # exchange among soil pools, slow pool to armored pool
    Te_exsa = input_dict.get('Te_exsa', 0.5)

# Armored terrestrial
    # evasion due to respiration of organic carbon
    E_Te_ra = input_dict.get('Te_ra', tot_resp*25/735)
    # biomass burning, protected pool to atmosphere
    E_Te_bba = input_dict.get('Te_bba', tot_bb*(fsoil*fCarmored) )
    # exchange among soil pools, armored pool to fast pool
    Te_exaf = input_dict.get('Te_exaf', 15)
    # exchange from armored pool to mineral pool
    Te_exam = input_dict.get('Te_exam', 0)

    # Budget closing term representing photoreduction etc
    E_Te_p = input_dict.get('Te_p', 0.01)

# Rivers -- 
    IHgD_pristine = input_dict.get('IHgD_pristine', 78.)
    IHgP_pristine = input_dict.get('IHgP_pristine', 659.)

    # total discharged to ocean margins
    Te_riv_margin = input_dict.get('Te_riv_margin', IHgD_pristine + IHgP_pristine)

    # global fraction of riverine HgP reaching the open oceans (Table 2, Amos et al. 2014)
    Lriver_FHgP = 'WeightedWalsh'
    if Lriver_FHgP == 'Walsh':
        f_HgPexport = 0.30
    elif Lriver_FHgP == 'Chester':
        f_HgPexport = 0.10
    elif Lriver_FHgP == 'WeightedWalsh':
        f_HgPexport = 0.07
    f_HgPexport = input_dict.get('f_HgPexport', f_HgPexport)

    # total reaching the open ocean
    Te_riv_ocean = input_dict.get('Te_riv_ocean', IHgD_pristine + f_HgPexport*IHgP_pristine)

    # Riverine discharge of terrestrial Hg to ocean margins
    T_riv_f = input_dict.get('T_riv_f', (Te_riv_margin*fveg + (Te_riv_margin*fsoil*fCfast)) )
    T_riv_s = input_dict.get('T_riv_s', (Te_riv_margin*fsoil*fCslow) )
    T_riv_a = input_dict.get('T_riv_a', (Te_riv_margin*fsoil*fCarmored) )

    # Riverine discharge of terrestrial Hg to open ocean
    O_riv_f = input_dict.get('O_riv_f', (Te_riv_ocean*fveg + (Te_riv_ocean*fsoil*fCfast)) )
    O_riv_s = input_dict.get('O_riv_s', (Te_riv_ocean*fsoil*fCslow) )
    O_riv_a = input_dict.get('O_riv_a', (Te_riv_ocean*fsoil*fCarmored) )

    assert np.abs((T_riv_f + T_riv_s + T_riv_a) - Te_riv_margin) < 1e-6, 'T_riv_f + T_riv_s + T_riv_a must equal Te_riv_margin'
    assert np.abs((O_riv_f + O_riv_s + O_riv_a) - Te_riv_ocean)  < 1e-6, 'O_riv_f + O_riv_s + O_riv_a must equal Te_riv_ocean'

# --------------------------------------------------------------------------
# Atmospheric rates (1/year)
# --------------------------------------------------------------------------
    k_A_oHgII = Dep_oHgII / Ratm
    model.add_process(Process('k_A_oHgII', 1/k_A_oHgII, compartment_from='atm', compartment_to='ocs',
                              reference='HgII deposition to surface ocean'
                              ))
    k_A_oHg0 = Upt_oHg0 / Ratm
    model.add_process(Process('k_A_oHg0', 1/k_A_oHg0, compartment_from='atm', compartment_to='ocs',
                              reference='gross Hg0 uptake by the surface ocean'
                              ))
    k_A_tHg0 = Dep_tHg0 / Ratm   # Hg0  deposition to terrestrial surfaces (largely vegetative uptake)
    model.add_process(Process('k_A_tfHg0', 1/k_A_tHg0, compartment_from='atm', compartment_to='tf',
                              reference='Hg0  deposition to terrestrial surfaces;Assumed to represent terrestrial leaf uptake'
                              ))
    
    k_A_tHgII = Dep_tHgII / Ratm
    # fraction of atmopsheric deposition to...
    # the fast soil pool
    fdep_tf = input_dict.get('fdep_tf', 0.5027)
    # the slow soil pool
    fdep_ts = input_dict.get('fdep_ts', 0.3213)
    # the fast armored pool
    fdep_ta = input_dict.get('fdep_ta', 0.1760)
    model.add_process(Process('k_A_tfHgII', 1/(k_A_tHgII*fdep_tf), compartment_from='atm', compartment_to='tf',
                              reference='HgII deposition to terrestrial fast pool'
                              ))
    model.add_process(Process('k_A_tsHgII', 1/(k_A_tHgII*fdep_ts), compartment_from='atm', compartment_to='ts',
                              reference='HgII deposition to terrestrial slow pool'
                              ))
    model.add_process(Process('k_A_taHgII', 1/(k_A_tHgII*fdep_ta), compartment_from='atm', compartment_to='ta',
                              reference='HgII deposition to terrestrial armored pool'
                              ))

# --------------------------------------------------------------------------
# Surface ocean rates (1/year)
# --------------------------------------------------------------------------
    k_Oc_ev = Ev_Hg0_ocs / Rocs
    model.add_process(Process('k_Oc_ev', 1/k_Oc_ev, compartment_from='ocs', compartment_to='atm',
                              reference='evasion Hg0 (gross flux)'
                              ))
    k_Oc_sp1 = ps_ocs / Rocs
    model.add_process(Process('k_Oc_sp1', 1/k_Oc_sp1, compartment_from='ocs', compartment_to='oci',
                              reference='particle settling; surface to intermediate'
                              ))
    k_Oc_vsi = vert_ocsi / Rocs
    model.add_process(Process('k_Oc_vsi', 1/k_Oc_vsi, compartment_from='ocs', compartment_to='oci',
                              reference='gross detrainment to intermediate ocean'
                              ))

# --------------------------------------------------------------------------
# Intermediate ocean rates (1/year)
# --------------------------------------------------------------------------
    k_Oc_sp2 = ps_oci / Roci
    model.add_process(Process('k_Oc_sp2', 1/k_Oc_sp2, compartment_from='oci', compartment_to='ocd',
                              reference='particle settling; intermediate to deep'
                              ))
    k_Oc_vis = vert_ocis / Roci
    model.add_process(Process('k_Oc_vis', 1/k_Oc_vis, compartment_from='oci', compartment_to='ocs',
                              reference='vertical seawater flow, intermediate to surface'
                              ))
    k_Oc_vid = vert_ocid / Roci
    model.add_process(Process('k_Oc_vid', 1/k_Oc_vid, compartment_from='oci', compartment_to='ocd',
                              reference='vertical seawater flow and diffusion, intermediate to deep'
                              ))

# --------------------------------------------------------------------------
# Deep ocean rates (1/year)
# --------------------------------------------------------------------------
    k_Oc_sp3 = ps_ocd / Rocd
    model.add_process(Process('k_Oc_sp3', 1/k_Oc_sp3, compartment_from='ocd', compartment_to=None,
                              reference='particle settling, deep ocean burial in deep sediments'
                              ))
    k_Oc_vdi = vert_ocdi / Rocd
    model.add_process(Process('k_Oc_vdi', 1/k_Oc_vdi, compartment_from='ocd', compartment_to='oci',
                              reference='vertical seawater flow and diffusion, deep to intermediate'
                              ))

# --------------------------------------------------------------------------
# Fast terrestrial reservoir rates (1/year)
# --------------------------------------------------------------------------

# Includes vegetation, ice, and fast+intermediate carbon pools from
# Smith-Downey et al. (2010)

    k_Te_rf = E_Te_rf / Rtf
    model.add_process(Process('k_Te_rf', 1/k_Te_rf, compartment_from='tf', compartment_to='atm',
                              reference='respiration; fast pool'
                              ))
    k_Te_p = E_Te_p / Rtf
    model.add_process(Process('k_Te_p', 1/k_Te_p, compartment_from='tf', compartment_to='atm',
                              reference='photoreduction and re-release of deposited Hg0'
                              ))
    k_Te_bbf = E_Te_bbf / Rtf
    model.add_process(Process('k_Te_bbf', 1/k_Te_bbf, compartment_from='tf', compartment_to='atm',
                              reference='biomass burning; fast pool'
                              ))

    k_T_exfs = Te_exfs / Rtf
    model.add_process(Process('k_T_exfs', 1/k_T_exfs, compartment_from='tf', compartment_to='ts',
                              reference='exchange among soil pools, fast pool to slow pool'
                              ))
    k_T_exfa = Te_exfa / Rtf
    model.add_process(Process('k_T_exfa', 1/k_T_exfa, compartment_from='tf', compartment_to='ta',
                              reference='exchange among soil pools, fast pool to armored pool'
                              ))

# --------------------------------------------------------------------------
# Slow terrestrial reservoir rates (1/year)
# --------------------------------------------------------------------------

    k_Te_rs = E_Te_rs / Rts
    model.add_process(Process('k_Te_rs', 1/k_Te_rs, compartment_from='ts', compartment_to='atm',
                              reference='evasion due to respiration of organic carbon; slow pool'
                              ))
    k_Te_bbs = E_Te_bbs / Rts
    model.add_process(Process('k_Te_bbs', 1/k_Te_bbs, compartment_from='ts', compartment_to='atm',
                              reference='biomass burning; slow pool'
                              ))
    k_T_exsf = Te_exsf / Rts
    model.add_process(Process('k_T_exsf', 1/k_T_exsf, compartment_from='ts', compartment_to='tf',
                              reference='exchange among soil pools, slow pool to fast pool'
                              ))
    k_T_exsa = Te_exsa / Rts
    model.add_process(Process('k_T_exsa', 1/k_T_exsa, compartment_from='ts', compartment_to='ta',
                              reference='exchange among soil pools, slow pool to armored pool'
                              ))

# --------------------------------------------------------------------------
# Armored terrestrial reservoir rates (1/year)
# --------------------------------------------------------------------------

    k_Te_ra = E_Te_ra / Rta
    model.add_process(Process('k_Te_ra', 1/k_Te_ra, compartment_from='ta', compartment_to='atm',
                              reference='evasion due to respiration of organic carbon; protected pool'
                              ))
    k_Te_bba = E_Te_bba / Rta
    model.add_process(Process('k_Te_bba', 1/k_Te_bba, compartment_from='ta', compartment_to='atm',
                              reference='biomass burning; protected pool'
                              ))
    k_T_exaf = Te_exaf / Rta
    model.add_process(Process('k_T_exaf', 1/k_T_exaf, compartment_from='ta', compartment_to='tf',
                              reference='exchange among soil pools, protected pool to fast pool'
                              ))

# --------------------------------------------------------------------------
# Rivers
# --------------------------------------------------------------------------

# First-order rate coefficients (1/yr)
    # Riverine discharge of terrestrial Hg to ocean margins
    k_T_riv_f = T_riv_f / Rtf  # fast
    k_T_riv_s = T_riv_s / Rts  # slow
    k_T_riv_a = T_riv_a / Rta  # armored

    # Riverine discharge of terrestrial Hg to open ocean
    k_O_riv_f = O_riv_f / Rtf
    k_O_riv_s = O_riv_s / Rts
    k_O_riv_a = O_riv_a / Rta

    model.add_process(Process('k_O_riv_f', 1/k_O_riv_f, compartment_from='tf', compartment_to='ocs',
                              reference='riverine discharge of terrestrial Hg to open ocean; fast pool'
                              ))
    model.add_process(Process('k_O_riv_s', 1/k_O_riv_s, compartment_from='ts', compartment_to='ocs',
                              reference='riverine discharge of terrestrial Hg to open ocean; slow pool'
                              ))
    model.add_process(Process('k_O_riv_a', 1/k_O_riv_a, compartment_from='ta', compartment_to='ocs',
                              reference='riverine discharge of terrestrial Hg to open ocean; armored pool'
                              ))

    k_L_riv_f = k_T_riv_f - k_O_riv_f
    model.add_process(Process('k_L_riv_f', 1/k_L_riv_f, compartment_from='tf', compartment_to=None,
                              reference='riverine discharge of terrestrial Hg to ocean margin sediment; fast pool'
                              ))
    k_L_riv_s = k_T_riv_s - k_O_riv_s
    model.add_process(Process('k_L_riv_s', 1/k_L_riv_s, compartment_from='ts', compartment_to=None,
                              reference='riverine discharge of terrestrial Hg to ocean margin sediment; slow pool'
                              ))
    k_L_riv_a = k_T_riv_a - k_O_riv_a
    model.add_process(Process('k_L_riv_a', 1/k_L_riv_a, compartment_from='ta', compartment_to=None,
                              reference='riverine discharge of terrestrial Hg to ocean margin sediment; armored pool'
                              ))
    
    #--------------------------------------------------------------------------
    # Fast waste pool
    #--------------------------------------------------------------------------
    k_We_wf = (k_Te_rf+k_Te_p+k_Te_bbf)
    model.add_process(Process('k_We_wf', 1/k_We_wf, compartment_from='wf', compartment_to='atm', 
                        reference='volatilization; waste fast',
                       ))
    
    ## option 2: "leaching" has same flow scheme as equivalent terrestrial pool
    k_Wl_wf_ts   = k_T_exfs
    model.add_process(Process('k_Wl_wf_ts', 1/k_Wl_wf_ts, compartment_from='wf', compartment_to='ts', 
                        reference='leaching; waste fast to slow terrestrial'
                        ))
    k_Wl_wf_ta   = k_T_exfa
    model.add_process(Process('k_Wl_wf_ta', 1/k_Wl_wf_ta, compartment_from='wf', compartment_to='ta', 
                        reference='leaching; waste fast to armored terrestrial'
                        ))
    k_Wl_wf_O = k_O_riv_f
    model.add_process(Process('k_Wl_wf_O', 1/k_Wl_wf_O, compartment_from='wf', compartment_to='ocs', 
                        reference='leaching --> riverine discharge; waste fast to surface ocean'
                        ))
    k_Wl_wf_riv_L = k_L_riv_f
    model.add_process(Process('k_Wl_wf_riv_L', 1/k_Wl_wf_riv_L, compartment_from='wf', compartment_to=None, 
                        reference='leaching --> riverine discharge; waste fast to ocean margin sediment'
                        ))

    #--------------------------------------------------------------------------
    # Slow waste pool
    #--------------------------------------------------------------------------
    k_We_ws    = k_Te_rs+k_Te_bbs       
    model.add_process(Process('k_We_ws', 1/k_We_ws, compartment_from='ws', compartment_to='atm', 
                        reference='volatilization; waste slow'
                        ))
        
    k_Wl_ws_tf = k_T_exsf       
    model.add_process(Process('k_Wl_ws_tf', 1/k_Wl_ws_tf, compartment_from='ws', compartment_to='tf', 
                        reference='leaching; waste slow to fast terrestrial'
                        ))    
    k_Wl_ws_ta = k_T_exsa       
    model.add_process(Process('k_Wl_ws_ta', 1/k_Wl_ws_ta, compartment_from='ws', compartment_to='ta', 
                        reference='leaching; waste slow to armored terrestrial'
                        ))
    k_Wl_ws_O = k_O_riv_s                          
    model.add_process(Process('k_Wl_ws_O', 1/k_Wl_ws_O, compartment_from='ws', compartment_to='ocs', 
                        reference='leaching --> riverine discharge; waste slow to surface ocean'
                        ))
    k_Wl_ws_riv_L = k_L_riv_s
    model.add_process(Process('k_Wl_ws_riv_L', 1/k_Wl_ws_riv_L, compartment_from='ws', compartment_to=None, 
                        reference='leaching --> riverine discharge; waste slow to ocean margin sediment'
                        ))
    #--------------------------------------------------------------------------
    # Permanent waste pool
    #--------------------------------------------------------------------------
    k_We_wa = k_Te_ra+k_Te_bba       
    model.add_process(Process('k_We_wa', 1/k_We_wa, compartment_from='wa', compartment_to='atm', 
                        reference='volatilization; waste armored'
                        ))
    k_Wl_wa_tf = k_T_exaf       
    model.add_process(Process('k_Wl_wa_tf', 1/k_Wl_wa_tf, compartment_from='wa', compartment_to='tf', 
                        reference='leaching; permanent waste to fast terrestrial'
                        ))
    k_Wl_wa_O = k_O_riv_a                       
    model.add_process(Process('k_Wl_wa_O', 1/k_Wl_wa_O, compartment_from='wa', compartment_to='ocs', 
                        reference='leaching --> riverine discharge; permanent waste to surface ocean'
                        ))
    k_Wl_wa_riv_L = k_L_riv_a
    model.add_process(Process('k_Wl_wa_riv_L', 1/k_Wl_wa_riv_L, compartment_from='wa', compartment_to=None, 
                        reference='leaching --> riverine discharge; permanent waste to ocean margin sediment'
                        ))

    model.build_matrix()


    return model
