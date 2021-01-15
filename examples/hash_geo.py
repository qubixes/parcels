from math import ceil

import numpy as np

from geo_utils import fast_distance
from base_neighbor import BaseNeighborSearch


class HashGeoNSearch(BaseNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    def __init__(self, values, max_dist):
        super(HashGeoNSearch, self).__init__(values, max_dist)
        self.build_tree()

    def find_neighbors(self, particle_id):
        hash_id = self._particle_hashes[particle_id]
        coor = self._values[:, particle_id]
        neighbor_blocks = geo_hash_to_neighbors(
            hash_id, coor, self._bits, self.max_dist)
        all_neighbor_points = []
        for block in neighbor_blocks:
            try:
                all_neighbor_points.extend(self._hashtable[block])
            except KeyError:
                pass

        true_neigh = []
        for neigh in all_neighbor_points:
            distance = fast_distance(
                *self._values[:, neigh], *self._values[:, particle_id])
            if distance < self.max_dist:
                true_neigh.append(neigh)
        return true_neigh, len(all_neighbor_points)

    def build_tree(self):
        '''Create the hashtable and related structures.'''
        epsilon = 1e-8
        n_lines_lat = int(ceil(np.pi/self.max_dist+epsilon))
        n_lines_long = int(ceil(2*np.pi/self.max_dist+epsilon))
        n_bits_lat = ceil(np.log(n_lines_lat)/np.log(2))
        n_bits_long = ceil(np.log(n_lines_long)/np.log(2))
        lat = self._values[0, :]
        long = self._values[1, :]

        lat_sign = (lat > 0).astype(int)
        i_lat = np.floor(np.abs(lat)/self.max_dist).astype(int)
        circ_small = 2*np.pi*np.cos((i_lat+1)*self.max_dist)
        n_long = np.floor(circ_small/self.max_dist).astype(int)
        n_long[n_long < 1] = 1
        d_long = 2*np.pi/n_long
        i_long = np.floor(long/d_long).astype(int)
        point_hash = np.bitwise_or(np.bitwise_or(
                lat_sign,
                np.left_shift(i_lat, 1)),
            np.left_shift(i_long, 1+n_bits_lat))

        self._hashtable = hash_split(point_hash)
        self._particle_hashes = point_hash
        self._bits = [n_bits_lat, n_bits_long]


def hash_split(hash_ids):
    '''Create a hashtable.

    Multiple particles that are found in the same cell are put in a list
    with that particular hash.
    '''
    sort_idx = np.argsort(hash_ids)
    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))


def i_lat_long_to_hash(i_lat, i_long, lat_sign, bits):
    '''Convert longitude and lattitude id's to hash'''
    point_hash = np.bitwise_or(np.bitwise_or(
        lat_sign,
        np.left_shift(i_lat, 1)),
        np.left_shift(i_long, 1+bits[0]))
    return point_hash


def geo_hash_to_neighbors(hash_id, coor, bits, max_dist):
    '''Compute the hashes of all neighboring cells.'''
    lat_sign = hash_id & 0x1
    i_lat = (hash_id >> 1) & ((1 << bits[0])-1)

    neighbors = []
    # Lower row, middle row, upper row
    for i_d_lat in [-1, 0, 1]:
        new_lat_sign = lat_sign
        new_i_lat = i_lat + i_d_lat
        if new_i_lat == -1:
            new_i_lat = 0
            new_lat_sign = (1-lat_sign)

        min_lat = new_i_lat + 1
        circ_small = 2*np.pi*np.cos(min_lat*max_dist)
        n_new_long = int(max(1, np.floor(circ_small/max_dist)))
        d_long = 2*np.pi/n_new_long
        if n_new_long <= 3:
            for new_i_long in range(n_new_long):
                new_hash = i_lat_long_to_hash(new_i_lat, new_i_long,
                                              new_lat_sign, bits)
                neighbors.append(new_hash)
        else:
            start_i_long = int(np.floor(coor[1]/d_long))
            for d_long in [-1, 0, 1]:
                new_i_long = (start_i_long+d_long+n_new_long) % n_new_long
                new_hash = i_lat_long_to_hash(new_i_lat, new_i_long,
                                              new_lat_sign, bits)
                neighbors.append(new_hash)
    return neighbors
