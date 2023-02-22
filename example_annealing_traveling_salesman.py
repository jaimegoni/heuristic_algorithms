# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:11:08 2020

@author: Jaime Goñi
"""
import math
import numpy as np

from heuristic_algorithms import simulated_annealing

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

s_ann = simulated_annealing()

s_ann.set_algorithm(objective = 'min', mix_type = 'traveling salesman')

s_ann.set_algorithm_values(temperature_function = 'default', initial_temp= 1000, max_bad_iterations= 50)

s_ann.set_variables(distances_matrix= distances_matrix)

s_ann.initialize_datasaving('./new_dir')

s_ann.iterate(n_iterations= 1000, graph_process= True)

s_ann.stop_datasaving()
print(s_ann.best_solution_found)