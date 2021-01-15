import numpy as np


def fast_distance(lat1, long1, lat2, long2):
    '''Compute the arc distance assuming the earth is a sphere.'''
    g = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)
    return np.arccos(np.minimum(1, g))
