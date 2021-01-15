import numpy as np

from geo_utils import fast_distance
from base_neighbor import BaseNeighborSearch


class BruteGeoNSearch(BaseNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    def find_neighbors(self, particle_id):
        distances = fast_distance(*self._values[:, particle_id],
                                  self._values[0, :], self._values[1, :])
        idx = np.where(distances < self.max_dist)[0]
        return idx
