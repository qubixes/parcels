from math import ceil

import numpy as np
from base_neighbor import BaseNeighborSearchCart
from numba import jit, njit
import numba as nb


class HashCartNSearch(BaseNeighborSearchCart):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    def __init__(self, values, max_dist):
        super(HashCartNSearch, self).__init__(values, max_dist)
        self._box = [[self._values[i, :].min(), self._values[i, :].max()]
                     for i in range(self._values.shape[0])]
        self.build_tree()

    def find_neighbors(self, particle_id):
        hash_id = self._particle_hashes[particle_id]
        neighbor_blocks = hash_to_neighbors(hash_id, self._bits)
        all_neighbor_points = []
        for block in neighbor_blocks:
            try:
                all_neighbor_points.extend(self._oct_dict[block])
            except KeyError:
                pass

        true_neigh = []
        for neigh in all_neighbor_points:
            dist = np.linalg.norm(
                self._values[:, neigh]-self._values[:, particle_id])
            if dist < self.max_dist:
                true_neigh.append(neigh)
        return true_neigh, len(all_neighbor_points)

    def build_tree(self):
        self._oct_dict, self._particle_hashes, self._bits = build_tree(
            self._values, self.max_dist, self._box)


def build_tree(values, max_dist, box):
    bits = []
    min_box = []
    for interval in box:
        epsilon = 1e-8
        n_bits = np.log((interval[1] - interval[0])/max_dist+epsilon)/np.log(2)
        bits.append(ceil(n_bits))
        min_box.append(interval[0])

    #min_box = np.array(min_box)
    #particle_hashes = np.empty(values.shape[1], dtype=int)
    #for particle_id in range(values.shape[1]):
    #    box_f = (values[:, particle_id] - min_box)/max_dist
    #    particle_hashes[particle_id] = np.bitwise_or(
    #        int(box_f[0]), np.left_shift(int(box_f[1]), bits[0]))
    min_box = np.array(min_box).reshape(-1, 1)
    box_i = ((values-min_box)/max_dist).astype(int)
    particle_hashes = np.bitwise_or(box_i[0, :],
                                    np.left_shift(box_i[1, :],
                                                  bits[0]))
    oct_dict = hash_split(particle_hashes)
    return oct_dict, particle_hashes, np.array(bits, dtype=int)


def hash_split(hash_ids):
    sort_idx = np.argsort(hash_ids)
    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate((np.array([True]), a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))


@njit
def hash_to_neighbors(hash_id, bits):
#   coor = np.zeros((len(bits),), dtype=np.int32)
#   new_coor = np.zeros((len(bits),), dtype=np.int32)
    coor = np.zeros((len(bits),), dtype=nb.int32)
    new_coor = np.zeros((len(bits),), dtype=nb.int32)
    tot_bits = 0
    for dim in range(len(bits)):
        coor[dim] = (hash_id >> tot_bits) & ((1 << bits[dim])-1)
        tot_bits += bits[dim]

    coor_max = np.left_shift(1, bits)

    neighbors = []

    for offset in range(pow(3, len(bits))):
        divider = 1
        for dim in range(len(bits)):
            new_coor[dim] = coor[dim] + (1-((offset//divider) % 3))
            divider *= 3
        if np.any(new_coor > coor_max) or np.any(new_coor < 0):
            continue
        new_hash = 0
        tot_bits = 0
        for dim in range(len(bits)):
            new_hash |= (new_coor[dim] << tot_bits)
            tot_bits += bits[dim]
        neighbors.append(new_hash)
    return neighbors
