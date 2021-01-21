import numpy as np

from base_neighbor import BaseNeighborSearchCart


class BruteCartNSearch(BaseNeighborSearchCart):
    def find_neighbors(self, particle_id):
        distances = np.linalg.norm(
            self._values - self._values[:, particle_id:particle_id+1],
            axis=0)
        idx = np.where(distances < self.max_dist)[0]
        return idx, self._values.shape[1]
