from abc import ABC, abstractmethod

import numpy as np
from timeit import timeit


class BaseNeighborSearch(ABC):
    def __init__(self, values, max_dist):
        self._values = values
        self.max_dist = max_dist

    @abstractmethod
    def find_neighbors(self, particle_id):
        '''Find neighbors with particle_id.'''
        pass

    def update_values(self, new_values):
        self._values = new_values

    @classmethod
    def benchmark(cls, density=1, max_n_particles=1000):
        '''Perform benchmarks to figure out scaling with particles.'''
        np.random.seed(213874)

        def bench_init(values, max_dist):
            return cls(values, max_dist)

        def bench_search(neigh_search):
            n_found = 0
            for i in range(neigh_search._values.shape[1]):
                n_found += len(neigh_search.find_neighbors(i))
            return n_found

        def create_positions(n_particles):
            yrange = 2*np.random.rand(n_particles)
            lat = np.arccos(1-yrange)-0.5*np.pi
            long = 2*np.pi*np.random.rand(n_particles)
            return np.array((lat, long))

        all_dt_init = []
        all_dt_search = []
        all_n_particles = []
        n_particles = 30
        n_init = 100
        n_search = 100
        while n_particles < max_n_particles:
            max_dist = 4*np.pi/(n_particles*density)
            if n_particles > 5000:
                n_init = 10
                n_search = 1
            positions = create_positions(n_particles)
            dt_init = timeit(lambda: bench_init(positions, max_dist),
                             number=n_init)/n_init
            neigh_search = bench_init(positions, max_dist)
            dt_search = timeit(lambda: bench_search(neigh_search),
                               number=n_search)/n_search
            all_dt_init.append(dt_init)
            all_dt_search.append(dt_search)
            all_n_particles.append(n_particles)
            n_particles *= 2
        return {"n_particles": np.array(all_n_particles),
                "init_time": np.array(all_dt_init),
                "search_time": np.array(all_dt_search)}
