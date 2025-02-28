import numpy as np
import xarray as xr

# Author: S. Eastham
# -- from latlontools.py in https://github.com/sdeastham/gcgridobj/

def latlon_extract(nc_file,force_poles=True):
    # Attempt to extract lat and lon data from a netCDF4 dataset
    lon_name = None
    lat_name = None
    nc_vars = nc_file.variables.keys()
    if 'lon' in nc_vars:
        lon_name = 'lon'
    elif 'longitude' in nc_vars:
        lon_name = 'longitude'
    else:
        raise ValueError('No longitude information found')
    if 'lat' in nc_vars:
        lat_name = 'lat'
    elif 'latitude' in nc_vars:
        lat_name = 'latitude'
    else:
        raise ValueError('No latitude information found')
    lon = np.ma.filled(nc_file[lon_name][:],0.0)
    lat = np.ma.filled(nc_file[lat_name][:],0.0)
    lon_b = latlon_est_bnds(lon)
    lat_b = latlon_est_bnds(lat,force_poles=force_poles)
    return lon_b, lat_b, lon, lat

def grid_area(lon_b=None, lat_b=None, hrz_grid=None, r_earth=None):

    if hrz_grid is not None:
       assert lon_b is None and lat_b is None, "Must provide either a grid object or both the latitude and longitude aedges"
       lon_b = hrz_grid['lon_b']
       lat_b = hrz_grid['lat_b']
    else:
       assert lon_b is not None and lat_b is not None, "Need both lon_b and lat_b if grid object not supplied"

    if r_earth is None:
       r_earth = 6.375e6 # Earth radius (m)

    # Calculate grid areas (m2) for a rectilinear grid
    lon_abs = []
    lastlon = lon_b[0]
    for i,lon in enumerate(lon_b):
        while lon < lastlon:
            lon += 360.0
        lon_abs.append(lon)
        lastlon = lon
   
    n_lat = lat_b.size - 1
    n_lon = lon_b.size - 1

    # Total surface area in each meridional band (allows for a regional domain)
    merid_area = 2*np.pi*r_earth*r_earth*(lon_abs[-1]-lon_abs[0])/(360.0*n_lon)
    grid_area = np.empty([n_lon,n_lat])
    lat_b_rad = np.pi * lat_b / 180.0
    for i_lat in range(n_lat):
        # Fraction of meridional area which applies
        sin_diff = np.sin(lat_b_rad[i_lat+1])-np.sin(lat_b_rad[i_lat])
        grid_area[:,i_lat] = sin_diff * merid_area

    # Transpose this - convention is [lat, lon]
    grid_area = np.transpose(grid_area)
    return grid_area

def latlon_est_bnds(indata,force_poles=False):
    # Estimate lat/lon edges based on a vector of mid-points
    dx = np.median(np.diff(indata))
    x0 = indata.data[0] - (dx/2.0)
    outdata = np.array([x0 + i*dx for i in range(0,indata.size + 1)])
    if force_poles:
        outdata[outdata<-90] = -90.0
        outdata[outdata>90] = 90.0
    return outdata

def latlon_est_mid(indata):
    # Calculate midpoints from edges
    return np.array([0.5*(indata[i] + indata[i+1]) for i in range(len(indata)-1)])


