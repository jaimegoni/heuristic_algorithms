# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 19:13:46 2020

@author: Jaime Goñi
"""

import math
import numpy as np

from heuristic_algorithms import genetic_adaptative_algorithm

def obj_fcn(input_vars):
    
    on = input_vars[0]
    mode = input_vars[1]
    phases = input_vars[2]
    amplitudes = input_vars[3]
    one_list = input_vars[4]
    
    if mode == 0:
        k = 0.25
    elif mode == 1:
        k = 0.52
    elif mode == 2:
        k = 0.88
    else:
        k = 1
        
    if on:
        aux = amplitudes * np.sin(phases)
    else:
        aux = amplitudes * np.cos(phases)
        
    loss = k*(np.mean(aux))
        
    return loss

variables = list()
variables.append({'variable':'on', 'initial_value': False, 'lower_limit':False, 'upper_limit' : True})
variables.append({'variable':'mode', 'initial_value': 0, 'lower_limit':0, 'upper_limit' : 3})
variables.append({'variable':'phases', 'initial_value': np.zeros(5).astype(float), 'lower_limit':0.0, 'upper_limit' : 2*math.pi})
variables.append({'variable':'amplitudes', 'initial_value': np.ones(5).astype(float), 'lower_limit':0.0, 'upper_limit' : 1.0})
variables.append({'variable':'one_list', 'initial_value': [0.0, 0.0, 0.0], 'lower_limit':0.0, 'upper_limit' : 2.0})

gen_adapt = genetic_adaptative_algorithm()
gen_adapt.set_algorithm(objective= 'max', mix_type= 'parameters', obj_function = obj_fcn)
gen_adapt.set_algorithm_values(pop_amount = 15)
gen_adapt.set_variables(variables)
gen_adapt.initialize_datasaving('./new_dir')
gen_adapt.get_initial_population()

for i in range(0, 5):
    gen_adapt.iterate()
    gen_adapt.graph_current_optimization_state()

gen_adapt.stop_datasaving()