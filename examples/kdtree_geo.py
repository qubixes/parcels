from math import ceil

import numpy as np


def fast_distance(lat1, long1, lat2, long2):
    g = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)
    return np.arccos(np.minimum(1, g))
        

def geo_brute_neighbors(i_val, values, max_dist=0.1):
    lat1 = values[0, i_val]
    long1 = values[1, i_val]
    distances = fast_distance(lat1, long1, values[0, :], values[1, :])
    idx = np.where(distances < max_dist)[0]
    return idx, values.shape[1]


def hash_split(hash_ids):
    sort_idx = np.argsort(hash_ids)
    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))


def lat_long_to_block(lat, long, max_dist):
    lat_sign = (lat>0).astype(int)
    i_lat = np.floor(np.abs(lat)/max_dist).astype(int)
    circ_small = 2*np.pi*np.cos((i_lat+1)*max_dist)
    n_long = np.floor(circ_small/max_dist).astype(int)
    n_long = np.maximum(1, n_long)
    d_long = 2*np.pi/n_long
    i_long = np.floor(long/d_long).astype(int)
    return lat_sign, i_lat, i_long, d_long, n_long


def i_lat_long_to_hash(i_lat, i_long, lat_sign, bits):
    point_hash = np.bitwise_or(np.bitwise_or(
        lat_sign,
        np.left_shift(i_lat, 1)),
    np.left_shift(i_long, 1+bits[0]))
    return point_hash


def geo_hash_to_neighbors(hash_id, coor, bits, max_dist):
    lat_sign = hash_id&0x1
    i_lat = (hash_id>>1)&((1<<bits[0])-1)
    i_long = hash_id>>(1+bits[0])

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
                new_hash = i_lat_long_to_hash(new_i_lat, new_i_long, new_lat_sign, bits)
                neighbors.append(new_hash)
        else:
            start_i_long = int(np.floor(coor[1]/d_long))
            for d_long in [-1, 0, 1]:
                new_i_long = (start_i_long+d_long+n_new_long)%n_new_long
                new_hash = i_lat_long_to_hash(new_i_lat, new_i_long, new_lat_sign, bits)
                neighbors.append(new_hash)
    return neighbors


def geo_find_neighbors(i_val, octo_dict, point_hashes, bits, values, max_dist=0.5):
    hash_id = point_hashes[i_val]
    coor = values[:, i_val]
    neighbor_blocks = geo_hash_to_neighbors(hash_id, coor, bits, max_dist)
    all_neighbor_points = []
    for block in neighbor_blocks:
        try:
            all_neighbor_points.extend(octo_dict[block])
        except KeyError:
            pass

    true_neigh = []
    for neigh in all_neighbor_points:
        if fast_distance(*values[:, neigh], *values[:, i_val]) < max_dist:
            true_neigh.append(neigh)
    return true_neigh, len(all_neighbor_points)


def geo_build_tree(values, max_dist=0.5):
    epsilon = 1e-8
    n_lines_lat = int(ceil(np.pi/max_dist+epsilon))
    n_lines_long = int(ceil(2*np.pi/max_dist+epsilon))
    n_bits_lat = ceil(np.log(n_lines_lat)/np.log(2))
    n_bits_long = ceil(np.log(n_lines_long)/np.log(2))
    lat = values[0, :]
    long = values[1, :]
    
    lat_sign = (lat>0).astype(int)
    i_lat = np.floor(np.abs(lat)/max_dist).astype(int)
    circ_small = 2*np.pi*np.cos((i_lat+1)*max_dist)
    n_long = np.floor(circ_small/max_dist).astype(int)
    n_long[n_long<1] = 1
    d_long = 2*np.pi/n_long
    i_long = np.floor(long/d_long).astype(int)
    point_hash = np.bitwise_or(np.bitwise_or(
            lat_sign,
            np.left_shift(i_lat, 1)),
        np.left_shift(i_long, 1+n_bits_lat))
    return hash_split(point_hash), point_hash, [n_bits_lat, n_bits_long]
