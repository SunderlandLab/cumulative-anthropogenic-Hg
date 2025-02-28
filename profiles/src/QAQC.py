import numpy as np
import pandas as pd

def apply_fixed_threshold(data, threshold: float, flag: str = 'exceeds fixed threshold'):
    ''' Apply a threshold to the data and flag the data that exceed the threshold. 
        This is intended to be used as a function for the assign_flags() method of the Profile() class.
        
        Parameters:
        -----------
        data: List[Datum]
            The data to be flagged.
        threshold: float
            The threshold value.
        flag: str
            The flag to be added to the data that exceed the threshold.
            
        Returns:
        --------
        List[Datum]
            The data with the flags added.
        
        Example:
        --------
        # p is a profile object
        p.assign_flags(fn=apply_fixed_threshold, threshold=0.5, flag='exceeds 0.5')
        '''
    for datum in data:
        if datum.value > threshold:
            datum.add_flag(flag)
    return data

def apply_std_outlier_detection(data, n_std: float = 3, flag: str = 'value greater than n standard deviations from the profile mean'):
    ''' Flag outliers based on the standard deviation of the profile. 
        This is intended to be used as a function for the assign_flags() method of the Profile() class.
        
        Parameters:
        -----------
        data: List[Datum]
            The data to be flagged.
        n_std: float
            The number of standard deviations from the mean to be considered an outlier.
        flag: str
            The flag to be added to the data that are outliers.
            
        Returns:
        --------
        List[Datum]
            The data with the flags added.
        
        Example:
        --------
        # p is a profile object
        p.assign_flags(fn=apply_std_outlier_detection, n_std=3, flag='value greater than 3 standard deviations from the profile mean')
    '''
    values = [datum.value for datum in data]
    mean = np.nanmean(values)
    std = np.nanstd(values)
    for datum in data:
        if datum.value > mean + n_std * std:
            datum.add_flag(flag)
    return data
