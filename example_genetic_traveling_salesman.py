# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:36:10 2020

@author: Jaime Goñi
"""

import math
import numpy as np

from heuristic_algorithms import genetic_algorithm

inf = math.inf

#Seven stops, starting and finishing at 1
#cities:             0   1   2   3    4    5    6    #cities:
#distances_matrix = [[0, 12, 10, inf, inf, inf,  12],  #0
#                    [0,  0,  8,  12, inf, inf, inf],  #1
#                    [0,  0,  0,  11,   3, inf,   9],  #2
#                    [0,  0,  0,   0,  11,  10, inf],  #3
#                    [0,  0,  0,   0,   0,   6,   7],  #4
#                    [0,  0,  0,   0,   0,   0,   9],  #5
#                    [0,  0,  0,   0,   0,   0,  0 ]]  #6


distances_matrix = np.random.randint(low = 5, high = 200, size = (70, 70))


genetic = genetic_algorithm()

genetic.set_algorithm(objective = 'min', mix_type = 'traveling salesman')

genetic.set_algorithm_values(pop_amount = 20)

genetic.set_variables(distances_matrix=distances_matrix, start_and_end_routes_at=0)

genetic.initialize_datasaving('./new_dir')
genetic.get_initial_population()

for i in range(0, 100):
    genetic.iterate()
    genetic.graph_current_optimization_state()

genetic.stop_datasaving()