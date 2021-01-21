from scipy.spatial import KDTree

from base_neighbor import BaseNeighborSearchCart


class ScipyCartNSearch(BaseNeighborSearchCart):
    def __init__(self, values, max_dist):
        super(ScipyCartNSearch, self).__init__(values, max_dist)
        self._kdtree = KDTree(values.T)

    def find_neighbors(self, particle_id):
        return self._kdtree.query_ball_point(self._values[:, particle_id], r=self.max_dist), 1
