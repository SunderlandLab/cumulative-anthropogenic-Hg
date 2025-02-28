import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import Akima1DInterpolator # requires scipy 1.13.0
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Union
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

import scipy
print('scipy version:', scipy.__version__)


@dataclass
class Datum():
    """A class to represent a point measurement of an individual oceanographic analyte."""

    quantity    : str   # name of the parameter
    depth       : float # depth [m] -- sea-level is 0
    lat         : float # latitude  [degrees N] (-90 to 90)
    lon         : float # longitude [degrees E] (-180 to 180)
    datetime    : pd.Timestamp # date and time of the measurement
    stationID   : str
    cruiseID    : str
    value       : float
    units       : str
    uncertainty : float = None
    uncertainty_type : str = None
    flags       : Union[List[str], None] = None #str = None
    reference   : str = None
    # -- default to none -- to be assigned later using classmethod
    basin       : str = None
    water_mass  : str = None
    # -- created in __post_init__ 
    year        : int = field(init=False)
    month       : int = field(init=False)
    day         : int = field(init=False)
    
    def convert_lon_360_to_180(self):
        """Convert longitude from 0-360 to -180-180."""
        self.lon = (self.lon + 180) % 360 - 180

    def __post_init__(self):
        self.year = self.datetime.year
        self.month = self.datetime.month
        self.day = self.datetime.day
        if self.flags is None:
            self.flags = []
        # convert longitude from 0-360 to -180-180
        self.convert_lon_360_to_180()

    def associate_basin_labels(self, ocean_basin_masks: xr.DataArray, mapping: Dict = None):
        """
        Associate ocean basin labels to the Datum based on its [lat, lon] attributes
        using an xarray DataArray of ocean basin masks. If a mapping is provided,
        the mapping will be used to convert the basin label to a different label.

        Parameters:
        ocean_basin_masks (xarray.DataArray): A DataArray containing ocean basin masks
                                               with dimensions [lat, lon].

        Returns:
        str: The ocean basin label associated with the Datum.

        TO-DO: Add support for [x, y, z] coordinates [e.g., water mass assignment using depth].
        """
        # check if the basin has already been assigned
        if self.basin is None:
            self.basin = ocean_basin_masks.sel(lat=self.lat, lon=self.lon, method='nearest').values.item()
            # apply mapping if provided (e.g., convert basin labels from integer to string name)
            if mapping is not None:
                self.basin = mapping.get(self.basin, self.basin)

        return self.basin
    
    def convert_units(self, conversion_factor: float, new_units: str, old_units: str = None):
        """Convert the value of the Datum to a new unit using the specified conversion factor."""
        # optional check that expected units match the current units
        if old_units is not None:
            if self.units != old_units:
                raise ValueError(f"Cannot convert units from {self.units} to {new_units} using conversion factor {conversion_factor}.")
        # convert the value to the new units and update the units attribute
        self.value = self.value * conversion_factor
        self.units = new_units

    def add_flag(self, flag:str):
        """Add a flag to the Datum."""
        if flag in self.flags:
            pass #print(f"Flag {flag} already exists for Datum {self}.")
        else:
            self.flags.append(flag)

@dataclass
class Profile():
    """A class to represent a vertical profile of oceanographic data (of a given analyte)."""

    quantity  : str # name of the parameter
    lat       : float # latitude  [degrees N]
    lon       : float # longitude [degrees E]
    datetime  : pd.Timestamp
    stationID : str
    cruiseID  : str
    depths    : ArrayLike # depths [m] -- sea-level is 0
    data      : List[Datum]
    units     : str
    reference : str = None
    horizontal_tolerance : float = 0.1 # tolerance for horizontal location equality

    def __post_init__(self):
        # sort data arrays by depth
        idx = np.argsort(self.depths)
        self.depths = self.depths[idx]
        self.data = [self.data[i] for i in idx]

        if not all([d.quantity == self.data[0].quantity for d in self.data]):
            raise ValueError("All data must be the same quantity.")
        if not all([self.same_horizontal_location(d) for d in self.data]):
            raise ValueError(f"All data must have the same latitude and longitude to within {self.horizontal_tolerance}.")
        #if not all([d.datetime == self.data[0].datetime for d in self.data]):
        #    raise ValueError("All data must have the same datetime.") # could do time tolerance
        if not all([d.stationID == self.data[0].stationID for d in self.data]):
            raise ValueError("All data must have the same stationID.")
        if not all([d.cruiseID == self.data[0].cruiseID for d in self.data]):
            raise ValueError("All data must have the same cruiseID.")
        if not all([d.units == self.data[0].units for d in self.data]):
            raise ValueError("All data must have the same units.")

        # remove nan data and associated depths
        self.depths = np.array([d.depth for d in self.data if not np.isnan(d.value)])
        self.data = [d for d in self.data if not np.isnan(d.value)]

        # construct uniqueID as the combination of cruiseID and stationID
        self.uniqueID = str(self.cruiseID) + '_' + str(self.stationID)
        
    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index: int):
        return self.data[index]
    
    def __repr__(self):
        return f"Profile(quantity={self.quantity}, lat={self.lat}, lon={self.lon}, datetime={self.datetime}, stationID={self.stationID}, cruiseID={self.cruiseID}, depths={self.depths}, data={self.data}, units={self.units} (n={len(self)})"

    def _latitude_tolerance(self, lat: float, datum: Datum):
        return abs(lat - datum.lat) <= self.horizontal_tolerance
    
    def _longitude_tolerance(self, lon: float, datum: Datum):
        return abs(lon - datum.lon) <= self.horizontal_tolerance
    
    def same_horizontal_location(self, datum: Datum):
        return (self._longitude_tolerance(self.lon, datum) and self._latitude_tolerance(self.lat, datum))

    # construct profile from list of Datum objects
    @classmethod
    def from_datums(cls, data: List[Datum], **kwargs):
        return cls(data[0].quantity, data[0].lat, data[0].lon, data[0].datetime,
                   data[0].stationID, data[0].cruiseID,
                   np.array([d.depth for d in data]), data, data[0].units,
                   data[0].reference, **kwargs)
    
    # construct profile from array of data
    @classmethod
    def from_arrays(cls, data: ArrayLike, depths: ArrayLike, quantity:str='default quantity',
                    lat:float=10, lon:float=10, datetime:pd.Timestamp=pd.Timestamp.now(),
                    stationID:str='default stationID', cruiseID:str='default cruiseID',
                    units:str='default units',
                    **kwargs):
    
        if len(data) != len(depths):
            raise ValueError("Data and depths must have the same length.")
        
        # unpack kwargs
        uncertainty = kwargs.get('uncertainty', [None for i in range(len(data))])
        uncertainty_type = kwargs.get('uncertainty_type', [None for i in range(len(data))])
        flags  = kwargs.get('flag', [None for i in range(len(data))])
        reference = kwargs.get('reference', [None for i in range(len(data))])

        datums = []
        for i in range(len(data)):
            datum = Datum(quantity=quantity, depth=depths[i], 
                          lat=lat, lon=lon, datetime=datetime, 
                          stationID=stationID, cruiseID=cruiseID, 
                          value=data[i], units=units, 
                          uncertainty=uncertainty[i], uncertainty_type=uncertainty_type[i], 
                          flags=flags[i], reference=reference[i],)
            
            datums.append(datum)
        
        return cls.from_datums(datums, **kwargs)
    
    # construct profile from DataFrame
    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, quantity_col: str = None, depth_col: str= 'depth', lat_col: str = 'lat', lon_col: str = 'lon', 
                       datetime_col: str = 'datetime', stationID_col: str = 'stationID', cruiseID_col: str = 'cruiseID',
                       value_col: str = 'value', units_col: str = 'units', uncertainty_col: str = 'uncertainty',
                       uncertainty_type_col: str = 'uncertainty_type', flag_col: str = 'flag', reference_col: str = 'reference',
                       **kwargs):
        # -- issue: generally when we read in a dataframe, the name of the analyte is specified in the column name.
        # Consider whether there is a better way to handle ingest of dataframes with multiple analytes as columns.
        if quantity_col is None:
            quantity_col = value_col # default to value column if quantity column not specified
        datums = []
        for index, row in data.iterrows():
            datum = Datum(quantity=quantity_col, depth=row[depth_col], lat=row[lat_col], lon=row[lon_col], 
                          datetime=row[datetime_col], stationID=row[stationID_col], cruiseID=row[cruiseID_col], 
                          value=row[value_col], units=row[units_col], uncertainty=row[uncertainty_col], 
                          uncertainty_type=row[uncertainty_type_col], flags=row[flag_col], reference=row[reference_col], 
                          )
            datums.append(datum)
        
        return cls.from_datums(datums, **kwargs)
    
    def to_dataframe(self):
        df = pd.DataFrame({'quantity': self.quantity, 'lat': self.lat, 'lon': self.lon, 
                           'datetime': self.datetime, 'stationID': self.stationID, 'cruiseID': self.cruiseID, 'uniqueID': self.uniqueID,
                           'depth': self.depths, 'value': [d.value for d in self.data], 'units': self.units, 
                           'uncertainty': [d.uncertainty for d in self.data], 'uncertainty_type': [d.uncertainty_type for d in self.data], 
                           'flags': [d.flags for d in self.data], 'reference': self.reference,
                           'year': [d.year for d in self.data], 'month': [d.month for d in self.data], 'day': [d.day for d in self.data],
                           'basin': [d.basin for d in self.data], 'water_mass': [d.water_mass for d in self.data]})
        return df
    
    def interpolate(self, depths_out: ArrayLike, method: str = 'linear', extrapolate: bool = False, exclude_flags: List = []):
        """Interpolate the profile to the specified depths using the specified method.
        
        Parameters:
        -----------
            
            depths_out (ArrayLike): 
                Depths to interpolate the profile to.

            method (str): 
                Interpolation method. Supported methods are ['linear', 'cubic', 'akima'].

            extrapolate (bool): 
                If True, extrapolate the data outside the range of the profile depths.

            exclude_flags (List):
                List of flags to exclude from the interpolation. If None, all flags are included.

        Returns:
        --------
            
            ArrayLike: 
                Interpolated data at depths_out.

        """
        for datum in self.data:
            if len(datum.flags)>0:
                print(f"Excluding flagged datum: {datum}")

        if (method == 'linear') and (extrapolate == True):
            data_out = np.interp(x=depths_out, 
                                 xp=[d.depth for d in self.data if not any(x in exclude_flags for x in d.flags)], 
                                 fp=[d.value for d in self.data if not any(x in exclude_flags for x in d.flags)])
            return data_out
        elif (method == 'linear') and (extrapolate == False):
            data_out = np.interp(x=depths_out, 
                                 xp=[d.depth for d in self.data if not any(x in exclude_flags for x in d.flags)],
                                 fp=[d.value for d in self.data if not any(x in exclude_flags for x in d.flags)], left=np.nan, right=np.nan)
            return data_out
        elif method == 'cubic': # cubic is generally a bad choice
            cs = CubicSpline(x=[d.depth for d in self.data if not any(x in exclude_flags for x in d.flags)],
                             y=[d.value for d in self.data if not any(x in exclude_flags for x in d.flags)], extrapolate=extrapolate)
            data_out = cs(depths_out)
            return data_out
        elif method == 'akima':
            # note that 'makima' method is used instead of 'akima'
            # makima does not overshoot the data, which is a big problem for cubic and a 
            # problem for akima
            cs = Akima1DInterpolator(x=[d.depth for d in self.data if not any(x in exclude_flags for x in d.flags)],
                                     y=[d.value for d in self.data if not any(x in exclude_flags for x in d.flags)], method="makima")
            data_out = cs(depths_out)
            return data_out
        else:
            raise ValueError("The following interpolation methods are supported: ['linear', 'cubic', 'akima']")

    def associate_basin_labels(self, ocean_basin_masks: xr.DataArray, mapping: Dict = None):
        """Extends the associate_basin_labels method to the Profile class 
           from the Datum method with the same name."""
        for datum in self.data:
            datum.associate_basin_labels(ocean_basin_masks, mapping)
    
    # -- assign flags to the profile
    # basic flag assignment functions to pass as `fn` argument are defined in QAQC.py
    def assign_flags(self, fn: Callable, **kwargs):
        """Assign flag to Datums according to a specified function."""
        self.data = fn(self.data, **kwargs)

    def convert_units(self, conversion_factor: float, new_units: str, old_units: str = None):
        """ Convert the units of all data in the Profile to a new unit using the specified conversion factor."""
        for datum in self.data:
            datum.convert_units(conversion_factor, new_units, old_units)
        self.units = new_units
        
@dataclass
class Collection():
    """A class to represent a collection of oceanographic data profiles."""
    
    profiles    : List[Profile]
    quantities  : List[str] = field(default_factory=list)
    latitudes   : List[float] = field(default_factory=list)
    longitudes  : List[float] = field(default_factory=list)
    datetimes   : List[pd.Timestamp] = field(default_factory=list)
    stationIDs  : List[str] = field(default_factory=list)
    cruiseIDs   : List[str] = field(default_factory=list)
    units       : List[str] = field(default_factory=list)
    references  : List[str] = field(default_factory=list)
    uniqueIDs   : List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.quantities = [p.quantity for p in self.profiles]
        self.latitudes  = [p.lat for p in self.profiles]
        self.longitudes = [p.lon for p in self.profiles]
        self.datetimes  = [p.datetime for p in self.profiles]
        self.stationIDs = [p.stationID for p in self.profiles]
        self.cruiseIDs  = [p.cruiseID for p in self.profiles]
        self.units      = [p.units for p in self.profiles]
        self.references = [p.reference for p in self.profiles]
        self.uniqueIDs  = [p.uniqueID for p in self.profiles]
        
    def __len__(self):
        return len(self.profiles)
    
    def __getitem__(self, index: int):
        return self.profiles[index]
    
    def __repr__(self):
        return f"Collection(n={len(self)}, quantities={self.quantities}, latitudes={self.latitudes}, longitudes={self.longitudes}, datetimes={self.datetimes}, stationIDs={self.stationIDs}, cruiseIDs={self.cruiseIDs}, units={self.units}, references={self.references})"
    
    def append(self, profile: Profile):
        self.profiles.append(profile)
        self.__post_init__()

    def append_collection(self, collection):
        # consider adding checks to prevent duplicate addition of profiles
        for profile in collection.profiles:
            self.profiles.append(profile)
        return self

    def to_dataframe(self):
        dfs = []
        for profile in self.profiles:
            df = profile.to_dataframe()
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def interpolate(self, depths_out: ArrayLike, method: str = 'linear', extrapolate: bool = False, exclude_flags: List = []):
        """ Interpolate all profiles in the collection to the specified depths and return a list of Profiles."""
        profiles_out = []
        for profile in self.profiles:
            # get length of profile.data after excluding flagged data
            n_non_nan_datums = len([d.value for d in profile.data if not np.isnan(d.value) and not any(x in exclude_flags for x in d.flags)])
            # if the profile is empty, skip it
            if n_non_nan_datums == 0:
                print(f"Skipping empty profile: {profile}")
                continue
            data_out = profile.interpolate(depths_out, method=method, extrapolate=extrapolate, exclude_flags=exclude_flags)
            profile_out = Profile.from_arrays(data_out, depths_out, quantity=profile.quantity, lat=profile.lat, lon=profile.lon, datetime=profile.datetime, stationID=profile.stationID, cruiseID=profile.cruiseID, units=profile.units)
            profiles_out.append(profile_out)
        return Collection(profiles_out)
    
    # associate basin labels to all profiles in the collection
    def associate_basin_labels(self, ocean_basin_masks: xr.DataArray, mapping: Dict = None):
        """ Associate ocean basin labels to all profiles in the Collection."""
        for profile in self.profiles:
            profile.associate_basin_labels(ocean_basin_masks, mapping)

    # subset by single attribute
    def subset_by_attribute(self, attribute: str, attribute_value):
        """ Return a new Collection object with profiles that match the specified attribute value."""
        profiles_out = [p for p in self.profiles if getattr(p, attribute) == attribute_value]
        return Collection(profiles_out)
    
    # subset by multiple attributes (passed as a dictionary)
    def subset_by_attributes(self, attributes: Dict):
        """ Return a new Collection object with profiles that match the specified attribute values."""
        profiles_out = self.profiles
        for attribute, attribute_value in attributes.items():
            profiles_out = [p for p in profiles_out if getattr(p, attribute) == attribute_value]
        return Collection(profiles_out)
    
    # update units where quantity matches
    def update_units(self, quantity: str, conversion_factor: float, new_units: str, old_units: str):
        """ Update the units of all profiles in the Collection where the old units quantity matches.
        
        Parameters:
        -----------
            quantity (str): 
                The quantity of the profiles to update.
            conversion_factor (float):
                The conversion factor to convert the profiles to the new units.
            new_units (str): 
                The new units to convert the profiles to.
            old_units (str): 
                The old units to convert the profiles from.
        """

        for profile in self.profiles:
            if (profile.quantity == quantity) & (profile.units == old_units):
                profile.convert_units(conversion_factor, new_units, old_units)

    #@classmethod
    #def from_dataframe(cls, data: pd.DataFrame, **kwargs):