import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# get key for which value contains `region_name`
def get_region(region_dict={}, country_name='Tonga'):
    '''get key for which value contains `country_name`
    
    Parameters
    ----------
    region_dict : dict
        dictionary of regions (keys) and their corresponding countries (values)
    region_name : str
        name of region to search for
    
    Returns
    -------
    str
        key of region_dict for which value contains `region_name`
        
    Examples
    --------
        get_region(region_dict=region_dict_7, region_name='Tonga')
        # returns 'Oceania'
    '''
    if len([k for k,v in region_dict.items() if country_name in v]) > 0:
        return [k for k,v in region_dict.items() if country_name in v][0]
    else:
        return None

region_dict_17 = {"Canada": ["Canada", "Saint Pierre and Miquelon"], 
                  "USA": ["United States of America", "United States Minor Outlying Islands"], 
                  "Central America": ["Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama", "Saint Martin", "Sint Maarten", "Haiti", "Dominican Republic", "US Naval Base Guantanamo Bay", "Cuba", "Aruba", "The Bahamas", "Turks and Caicos Islands", "Trinidad and Tobago", "Grenada", "Saint Vincent and the Grenadines", "Barbados", "Saint Lucia", "Dominica", "Puerto Rico", "Anguilla", "British Virgin Islands", "Jamaica", "Cayman Islands", "Bermuda", "Antigua and Barbuda", "Saint Kitts and Nevis", "Montserrat", "United States Virgin Islands", "Saint Barthelemy"], 
                  "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela", "Brazilian Island", "Cura\u00e7ao", "Southern Patagonian Ice Field", "South Georgia and the Islands", "Falkland Islands"], 
                  "Northern Africa": ["Algeria", "Egypt", "Libya", "Morocco", "Tunisia", "Western Sahara", "Gibraltar", "Bir Tawil"], 
                  "Western Africa": ["Benin", "Burkina Faso", "Cabo Verde", "Cameroon", "Chad", "Central African Republic", "Equatorial Guinea", "Ivory Coast", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Republic of the Congo", "Senegal", "Sierra Leone", "Togo", "S\u00e3o Tom\u00e9 and Principe"], 
                  "Eastern Africa": ["Burundi", "Democratic Republic of the Congo", "Kenya", "Rwanda", "South Sudan", "Uganda", "Djibouti", "Eritrea", "Ethiopia", "Somalia", "Comoros", "Madagascar", "Mauritius", "Seychelles", "Somaliland", "Sudan"], 
                  "Southern Africa": ["Angola", "Botswana", "eSwatini", "Lesotho", "Malawi", "Mozambique", "Namibia", "South Africa", "United Republic of Tanzania", "Zimbabwe", "Zambia"], 
                  "OECD Europe": ["Austria", "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Iceland", "Ireland", "Italy", "Luxembourg", "Netherlands", "Norway", "Portugal", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom", "Dhekelia Sovereign Base Area", "Liechtenstein", "San Marino", "Monaco", "Andorra", "Vatican", "Faroe Islands", "Greenland", "Saint Helena", "Malta", "Jersey", "Isle of Man", "Guernsey", "Aland", "Indian Ocean Territories"], 
                  "Eastern Europe": ["Albania", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Hungary", "Kosovo", "Moldova", "Montenegro", "North Macedonia", "Northern Cyprus", "Cyprus No Mans Area", "Poland", "Romania", "Republic of Serbia", "Slovakia", "Slovenia"], 
                  "Former USSR": ["Armenia", "Azerbaijan", "Belarus", "Estonia", "Georgia", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Moldova", "Russia", "Tajikistan", "Turkmenistan", "Ukraine", "Uzbekistan", "Baykonur Cosmodrome"], 
                  "Middle East": ["Akrotiri Sovereign Base Area", "Bahrain", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", "Palestine", "Qatar", "Saudi Arabia", "Syria", "United Arab Emirates", "Yemen"], 
                  "South Asia": ["Afghanistan", "Bangladesh", "British Indian Ocean Territory", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka", "Siachen Glacier"], 
                  "East Asia": ["China", "Hong Kong S.A.R.", "Macao S.A.R", "Mongolia", "North Korea", "South Korea", "Taiwan"], 
                  "Southeast Asia": ["Brunei", "Cambodia", "East Timor", "Indonesia", "Laos", "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand", "Vietnam"], 
                  "Oceania": ["Ashmore and Cartier Islands", "Australia", "Coral Sea Islands", "New Zealand", "Norfolk Island", "Fiji", "New Caledonia", "Papua New Guinea", "Solomon Islands", "Vanuatu", "Federated States of Micronesia", "Guam", "Kiribati", "Marshall Islands", "Nauru", "Northern Mariana Islands", "Palau", "American Samoa", "Cook Islands", "French Polynesia", "Samoa", "Tonga", "Tuvalu", "Niue", "Wallis and Futuna", "Heard Island and McDonald Islands"], 
                  "Japan": ["Japan"]}

region_mapping_17_to_4 = {'Americas': ['Canada','USA','Central America', 'South America'],
                          'Africa + Middle East': ['Northern Africa','Western Africa','Eastern Africa','Southern Africa', 'Middle East'],
                          'Europe + Former USSR': ['OECD Europe','Eastern Europe', 'Former USSR'],
                          'Asia + Oceania': ['South Asia','East Asia','Southeast Asia','Japan','Oceania'],}

region_dict_4 = {}
for region in region_mapping_17_to_4:
    region_dict_4[region] = []
    for subregion in region_mapping_17_to_4[region]:
        region_dict_4[region] += region_dict_17[subregion]

design_dict = {'Americas': {'color': '#0072BD', 'label': 'Americas'},
                'Africa + Middle East': {'color': '#D95319', 'label': 'Africa + Middle East'},
                'Asia + Oceania': {'color': [0.7,0.7,0.7], 'label': 'Asia + Oceania'},
                'Europe + Former USSR': {'color': '#EDB120', 'label': 'Europe + Former USSR'}}

# Get shapefiles of all countries from natural earth data
# http://www.naturalearthdata.com/downloads/10m-cultural-vectors/ne_10m_admin_0_countries/
countries = gpd.read_file('../inputs/misc/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
country_ids = {k: i for i, k in enumerate(countries.ADMIN)}
shapes = zip(countries.geometry, range(len(countries)))

# Create a new column in the countries GeoDataFrame for the region
countries['region'] = countries['SOVEREIGNT'].map(lambda x: get_region(region_dict=region_dict_4, country_name=x))

# dissolve the countries GeoDataFrame by region
regions = countries.dissolve(by='region')

# plot the regions
fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.Robinson()})
for region in regions.index:
    ax.add_geometries(regions[regions.index==region].geometry, crs=ccrs.PlateCarree(), facecolor=design_dict[region]['color'], edgecolor='black', lw=0.5, zorder=2)

# put ocean in the foreground
ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', edgecolor='k', lw=0.5, zorder=3)

# add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='0.5', linestyle='--', alpha=0.5, lw=0.5, zorder=1)

legend_elements = [Patch(facecolor=design_dict[key]['color'], edgecolor='black', lw=0.5, label=design_dict[key]['label']) for key in design_dict]
legend = ax.legend(handles=legend_elements, loc='lower center', fontsize=10, facecolor='white', framealpha=1, edgecolor='black', ncols=4, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.1))
legend.get_frame().set_linewidth(0.5)

plt.savefig('../figures/figure_S3.png', dpi=1200, bbox_inches='tight')
