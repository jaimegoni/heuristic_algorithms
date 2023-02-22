# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:47:10 2020

@author: Jaime Goñi
"""

import os
import csv
import math
import time
import copy
import scipy.stats as st
import numpy as np
import matplotlib as plt

class random_generator:
    """Class which develops some functions of the python "random" module, but
    actually with real random numbers"""
    
    float_precission = 9
    
    def __init__ (self, float_precission = 9):
        self.float_precission = float_precission
        
        return
    
    def random(self):
        number = int.from_bytes(os.urandom(2), byteorder = 'big') / (256**2)
        
        return round(number, self.float_precission)
        
    
    def choice(self, case_A, case_B = 0):
        
        if type(case_A) == list:
            l = len(case_A)
            chosen = self.randint(0, l-1)
            return case_A[chosen]
        
        else:
            if self.random()>=0.5:
                return case_A
            else:
                return case_B
        
        return
        
    def randint(self, low, up):
        
        number = self.random()
        number = low + number*(up-low)
        
        if self.random()>0.5:
            return int(math.floor(number))
        else:
            return int(math.ceil(number))
        
        return
    
    def uniform(self, low, up):
        number = self.random()
        number = low + number*(up-low)
        
        return number
    
    def gauss(self, mu, sigma):
        distribution = st.norm(mu, sigma)
        rand_normal_number = distribution.ppf(self.random())
        
        return rand_normal_number

class heuristic_algorithm:
    
    random = None
    
    objective = ''
    objective_function = None
    create_random_solution = None
    
    mix_type = ''
    
    var_names = list()
    parameters = list()                 #Parameters which define a member of the population (genes)
    
    destinies    = list()
    choosing_destinies = list()
    distances_matrix = None
    initial_route_point = 0
    
    best_solution_found = None
    
    float_precission = 9
    
    #FALTA: almacenar esta información
    already_working = False
    initial_time = 0
    actual_iteration = 0
    time_history = list()
    iterations_history = list()
    losses_history = list()
    
    losses_figure = None
    axs = None
    
    info_dir = ''
    info_file_name = ''
    info_writer = None
    save_info = False
    save_all = False
    closed_files = True
    parameter_files = list()
    
    def __init__ (self):
        
        return
    
    def __del__(self):
        
        if not(self.closed_files) and (self.save_info):
            self.stop_datasaving()
        
        return
    
    def set_algorithm(self, objective, mix_type, obj_function = None, float_precission= 9):
        """Sets the main parameters of the heuristic algorithm"""
        self.float_precission = float_precission
        self.random = random_generator(float_precission = float_precission)
        self.objective = objective
        self.mix_type = mix_type
        
        if mix_type == 'parameters':
            self.create_random_solution = self.create_random_parameter_solution
            if not(obj_function == None):
                self.set_objective_func(obj_function)
            else:
                print('>> ERROR: no objective function specified for the "parameters" optimization')
                print('>>> Please, specify an objective function')
                
                return False
            
        elif mix_type == 'traveling salesman':
            self.create_random_solution = self.create_random_traveling_solution
            if not(obj_function == None):
                self.set_objective_func(obj_function)
            else:
                self.set_objective_func(self.traveling_salesman_obj_function)
            
        else:
            print('>> ERROR: non identified mix type')
            print('>>> Please, choose between "parameters" or "traveling salesman"')
            
            return False
        
        return
    
    def set_objective_func(self, h):
        
        self.objective_function = h
        
        return
    
    def set_variables(self, input_params = {'variable':'var', 'initial_value': None, 'lower_limit':0, 'upper_limit' : 0}, distances_matrix = None, start_and_end_routes_at = 0):
        """Input values: dictionary of one parameter of list of dictionaries with parameters; dtype: float, int or boolean
         which influence the objective function and are the "genes" of a being. Symmetric matrix of distances
         between each point for the tavelling salesman problem (as numpy array)
         Limits included are closed at the end [], not open [)
        
         input_params can be either a dictionary or a list of dictionaries with the indicated keywords
        
         'initial_value' can be a number, a boolean, a list or a numpy array"""
        
        if self.mix_type == 'parameters':
            #If the input variables are storaged in a list
            
            params = list()
            
            if type(input_params) == list:
                for parameter in input_params:
                    self.var_names.append(parameter.get('variable'))
                    try:
                        del parameter['variable']
                    except:
                        pass
                    params.append(parameter)
                    
                self.parameters.extend(params)
            else:
                #Case only one variable is being included
                self.var_names.append(input_params.get('variable'))
                del input_params['variable']
                self.parameters.append(input_params)
            
        
        elif self.mix_type == 'traveling salesman':
            
            if type(distances_matrix) == list:
                distances_matrix = np.array(distances_matrix)
            
            #Check if the matrix is symmetric
            if (not(np.allclose(distances_matrix, distances_matrix.T, rtol= 1e-05, atol= 1e-08))):
                dims = distances_matrix.shape
                
                for i in list(range(0, dims[0])):
                    for j in list(range(0, dims[1])):
                        if (i < j):
                            distances_matrix[j, i] = distances_matrix[i, j]
                        elif j == i:
                            distances_matrix[j, i] = 0
            
            self.distances_matrix = distances_matrix
            self.destinies  = list(range(distances_matrix.shape[0]))
            self.initial_route_point = start_and_end_routes_at
            self.choosing_destinies = copy.deepcopy(self.destinies)
            self.choosing_destinies.remove(start_and_end_routes_at)
        
        else:
            print('Error: not programmed side of the code')
            print('Please choose between "parameters" or "traveling_salesman"')
            
        return
    
    def remove_variable(self, parameter_name):
        if self.mix_type == 'parameters':
            rem = self.var_names.index(parameter_name)
            self.var_names.remove(parameter_name)
            self.parameters.remove(self.parameters[rem])
        else:
            self.distances_matrix = None
            self.destinies = list()
        return
    
    def create_random_parameter_solution(self):
        
        solution = list()
        
        for parameter in self.parameters:
            #Fills the solution with random parameters/ genes
            
            fill = copy.copy(parameter.get('initial_value'))
            up = parameter.get('upper_limit')
            low = parameter.get('lower_limit')
            
            if up == math.inf:
                up = abs(low * 100)
            elif low == -math.inf:
                low = - abs(up * 100)
            
            if type(fill) == list:
                aux = copy.copy(fill)
                #If the gene is a list of floats, integers or booleans
                for pos in list(range(0, len(aux))):
                    aux[pos]=self.return_random_uniform(aux[pos], up, low)
                    
                solution.append(copy.deepcopy(aux))
                
            elif type(fill) == np.ndarray:
                #If the gene is a numpy array
                fill = np.random.uniform(low = low, high = up, size = fill.shape)
                solution.append(fill)
                
            else:
                #If the gene is an integer, a float or a boolean
                solution.append(self.return_random_uniform(fill, up, low))
               
                
            del (fill)
                
        solution.append(self.objective_function(solution))
        
        return copy.deepcopy(solution)
    
    def create_random_traveling_solution(self):
        """Creates a random solution (type: list) by filling """
        
        solution = list()
        
        scape = False
        
        while not(scape):
            #Loads the allowed destinies
            possible_destinies = copy.deepcopy(self.choosing_destinies)
            
            #Solution starts at the initial point
            solution.append(self.initial_route_point)
            
            while_loop_checker = 0
            
            while( len(possible_destinies) > 0):
                
                #Randomly choses the next step of the journey
                next_destiny = self.random.choice(possible_destinies)
                
                #Checks if it is an allowed destiny
                if (self.distances_matrix[solution[-1], next_destiny] != math.inf):
                    #If so, appends that destiny to the route and removes it from the possible ones
                    solution.append(next_destiny)
                    possible_destinies.remove(next_destiny)
                    
                    while_loop_checker = 0
                    
                else:
                    while_loop_checker +=1
                
                #Checks whether the loop is trapped in a dead end
                if (while_loop_checker > len(possible_destinies)):
                    #If so, restarts the loop
                    solution = list()
                    
                    #Loads the allowed destinies
                    possible_destinies = copy.deepcopy(self.choosing_destinies)
                    
                    #Solution starts at the initial point
                    solution.append(self.initial_route_point)
                    
                    while_loop_checker = 0
                    
    #            print(solution)
            
            #Route finishes at the initial point
            solution.append(self.initial_route_point)
            
            distance = self.objective_function(solution)
            
            if distance < math.inf:
                scape = True
#            else:
#                print('inf')
        
        check = set(solution)
        
        if not(len(check) == len(self.destinies)):
            print('Problema!! solución inicial encontrada no está bien')
            
        child = [solution, self.objective_function(solution)]
        
        return copy.deepcopy(child)
    
    def traveling_salesman_obj_function(self, route):
        
        total_distance = 0
        
        actual_location = route[0]
        
        for step in route[1:]:
            total_distance += self.distances_matrix[actual_location, step]
            actual_location = step
        
        return total_distance
    
    def return_random_uniform(self, value, up_lim, down_lim):
        """Returns a random value depending on the type of input 'value' """
#        print('type of input value is : '+str(type (value)), ' and boundaries are: ', up_lim, ',', down_lim)
        if (type (value) == int) or (type (value) == np.int32):
            return self.random.randint(down_lim, up_lim)
        
        elif type (value) == bool:
            return self.random.choice(up_lim, down_lim)
        
        else:
            return self.random.uniform(down_lim, up_lim)
        
        return
    
    def return_random_gauss(self, mu, sigma):
        """Returns a random value depending on the type of input 'value' """
#        print('type of input value is : '+str(type (value)), ' and boundaries are: ', up_lim, ',', down_lim)
        if (type (mu) == int) or (type (mu) == np.int32):
            value = self.random.gauss(mu = mu, sigma = sigma)
            
            while abs(value) == math.inf:
                value = self.random.gauss(mu = mu, sigma = sigma)
            
            if (self.random.random() > 0.5):
                value = int(math.ceil(value))
            else:
                value = int(math.floor(value))
                
            return value
        
        elif type (mu) == bool:
            return self.random.choice(True, False)
        
        else:
            return self.random.gauss(mu = mu, sigma = sigma)
        
        return
    
    def graph_current_optimization_state(self):
        
        if self.losses_figure == None:
            self.losses_figure = plt.pyplot.figure()
            self.axs = self.losses_figure.gca()
        
        
        self.axs.cla()
        self.axs.plot(self.iterations_history, self.losses_history)
        plt.pyplot.pause(0.05)
        
        return
    
    #FALTA: que guarde la mejor solución
    def initialize_datasaving(self, datasaving_directory, save_all = False):
        
        self.save_info = True
        self.closed_files = False
        self.save_all = save_all
        
        
        try:
            os.mkdir(datasaving_directory)
        except:
            pass
        
        actual_date = time.localtime()
        datetime_info =  '/exec_' + self.__class__.__name__ +'_DAY_' + str(actual_date.tm_mday) + '_' + str(actual_date.tm_mon) + '_H_' + str(actual_date.tm_hour)+ '_' + str(actual_date.tm_min)+ '_' + str(actual_date.tm_sec)
        datasaving_directory = datasaving_directory + datetime_info
        self.info_dir = datasaving_directory
        
        try:
            os.mkdir(datasaving_directory)
        except:
            pass
        
        self.info_file_name = datasaving_directory + '/algorithm_info.csv'
        self.info_file = open (self.info_file_name, mode = 'a', newline = '')
        self.info_writer = csv.writer(self.info_file, delimiter = ';')
        
        if self.__class__.__name__ == 'simulated_annealing':
            self.info_writer.writerow(['iteration', 'time (ms)', 'losses', 'temperature'])
        else:
            self.info_writer.writerow(['iteration', 'time (ms)', 'losses'])
        
        self.metadata_file_name = datasaving_directory + '/algorithm_metadata.txt'
        self.metadata_file = open (self.metadata_file_name, mode = 'a', newline = '')
        
        self.save_metadata()
        
        if save_all:
            
            for name in self.var_names:
                filename = datasaving_directory + '/' + name + '.csv'
                self.parameter_files.append(open (filename, mode = 'w', newline = ''))
        
        return
    
    def save_metadata(self, closing = False):
        
        return
    
    def save_best_parameter_solution_found(self):
        
        files = list()
        
        for name in self.var_names:
            filename = self.info_dir + '/best_result_' + name + '.csv'
            files.append(open (filename, mode = 'w', newline = ''))
        
        solution = self.best_solution_found[:-1]
        
        for parameter in list(range(0, len(solution))):
            
            self.save_parameter(files[parameter], solution[parameter])
        
        for file in files:
            file.close()
            
        return
    
    def save_best_route_found(self):
        filename = self.info_dir + '/best_route_found.txt'
        file = open(filename, mode = 'w', newline = '')
        
        solution = self.best_solution_found[0]
        
        text = ''
        
        for step in solution:
                text += str(step) + ';'
                
        file.write(text + '\n' )
        
        file.close()
        
        return
    
    def save_parameter(self, file, parameter):
        
        if type(parameter) == np.ndarray:
            np.savetxt(file, parameter, delimiter = ';')
            
            return
        
        elif type(parameter) == list:
            
            text = ''
            for value in parameter:
                text += str(value) + ';'
                
        else:
            text = str(parameter)
        
        writer = csv.writer(file, delimiter = ';')
        writer.writerow([text])
        
        return
    
    #FALTA: que guarde la mejor solución
    def save_optimization_info(self):
        
        self.info_writer. writerow([str(self.iterations_history[-1]), str(round(1000*(time.time()-self.initial_time))), str(self.losses_history[-1])])
        
        return
    
    #FALTA: que cierre TODOS los archivos
    def stop_datasaving(self):
        
        self.closed_files = True
        self.save_metadata(closing= True)
        
        if self.mix_type == 'parameters':
            self.save_best_parameter_solution_found()
        else:
            self.save_best_route_found()
        
        try:
            self.info_file.close()
            self.metadata_file.close()
            
            if self.save_all:
                for file in self.parameter_files:
                    file.close()
        except:
            pass
        
        self.save_info = False
        self.save_all = False
        
        print('>> All files have been closed')
        
        return

class genetic_algorithm(heuristic_algorithm):
                  
    population   = list()
    
    #Parameters of the genetic algorithm
    population_amount = 10                  #Higher amount recommended for big problems
    mutation_probability = 0.1              #Lower probability recommended for big problems
    
    best_parent_percentage = 0.4            #Percentage of the population of parents with best punctuation which will be coupled
    worst_parent_percentage = 0.2           #Percentage of the population of parents with worst punctuation which will be coupled
    
    generations_per_iteration = 10          #Number of generations computed by iteration of the algorithm
    
    
    pair_couple = None
    
    def __init__ (self):
        super().__init__()
        return
    
    def set_algorithm_values(self, pop_amount = 10, generations_per_iteration = 10, mutation_probab = 0.1, best_parent_selection = 0.4, worse_parent_selection = 0.2):
        
        self.population_amount = pop_amount
        self.generations_per_iteration = generations_per_iteration
        self.mutation_probability = mutation_probab
        self.best_parent_percentage = best_parent_selection
        self.worst_parent_percentage = worse_parent_selection
        
        return
    
    def set_objective_func(self, h):
        
        if self.mix_type == 'parameters':
            self.pair_couple = self.pair_copules_parameters
        elif self.mix_type == 'traveling salesman':
            self.pair_couple = self.pair_couples_routes
        else:
            print('>> ERROR: non identified mix type')
            print('>>> Please, choose between "parameters" or "traveling salesman"')
            
            return False
            
        self.objective_function = h
        
        return True
    
    def get_initial_population(self):
        
        self.population = list()
        
        for i in range(0, self.population_amount):
            self.population.append(self.create_random_solution())
        
        self.sort_population()
        
        return
    
    def get_random_population(self, pop_amount):
        
        new_population = list()
        
        for i in range(0, pop_amount):
            new_population.append(self.create_random_solution())
        
        return copy.deepcopy(new_population)
    
    def iterate(self):
        
        if not(self.already_working):
            self.initial_time = time.time()
            self.already_working = True
            
        for i in range(0, self.generations_per_iteration):
            self.mix_population()
            
            self.time_history.append(time.time() - self.initial_time)
            self.actual_iteration += 1
            self.iterations_history.append(self.actual_iteration)
            self.losses_history.append(self.population[0][-1])
            
            if self.save_info:
                self.save_optimization_info()
        
        self.sort_population()
        self.best_solution_found = self.population[0]
        
        return
    
    def mix_population(self):
        """Randomly choses the members of the population, couples them and returns
        a new generation, taking into account genetic algorithm's theory"""
        
        score = list()
        #looks for the scores of each member of the population
        for member in self.population:
            score.append(member[-1])
        #Indexes of mebers from lower to higher scores
        order = list(np.argsort(score))
        
        if self.objective == 'max':
            order.reverse()
            
        #Now, the order goes from the best punctuated to the worst
        
        del(score)
        
        pop = list()
        pairs = list()
        
        #Selects a 20% of the worst punctuated members of the population to be coupled
        w_num_to_choose = math.floor(self.population_amount*self.worst_parent_percentage)
        choose_members = order[-w_num_to_choose:]
        
        for pos in choose_members:
            pop.append(self.population[pos])
        
        #Selects a 40% of the best punctuated members of the population to be coupled
        b_num_to_choose = math.ceil(self.population_amount*self.best_parent_percentage)
        choose_members = order[:b_num_to_choose]
        
        for pos in choose_members:
            pop.append(self.population[pos])
            
        #We prefer an even number of parents, so we choose another member if necessary
        if len(pop)%2 !=0:
            pop.append(self.population[order[b_num_to_choose]])
        
        #Make random pairs
        ids = list(range(0,len(pop)))
        
        j = round(len(ids)/2)
        
        for i in range(0,j):
            
            chosen = self.random.choice(ids)
            parent_A = pop[chosen]
            ids.remove(chosen)
            
            chosen = self.random.choice(ids)
            parent_B = pop[chosen]
            ids.remove(chosen)
            
            pairs.append([parent_A, parent_B])
            del (parent_A, parent_B)
            
        children = list()
        
        #Each couple will have two kids
        pairs.extend(pairs)
        
        #Create the children
        for pair in pairs:
            children.append(self.pair_couple(pair[0], pair[1]))
        
        #Remove the worst members of the population in order to locate the children
        for i in range(0, len(children)):
            del(self.population[-1])
        
        #Add the children to the population
        self.population.extend(copy.deepcopy(children))
        
        del(children)
        
        return
    
    def pair_copules_parameters(self, parent_A, parent_B):
        child = list()
        gene = 0
        
        param_copy = copy.deepcopy(self.parameters)
        
        for parameter in param_copy:
            
            #Fills the child with parameters/ genes
            fill = copy.copy(parameter.get('initial_value'))
            up = parameter.get('upper_limit')
            low = parameter.get('lower_limit')
            
            if up == math.inf:
                up = abs(low * 100)
            elif low == -math.inf:
                low = - abs(up * 100)
                
            if type(fill) == list:
                #If the gene is a list of floats, integers or booleans
                aux = copy.copy(fill)
                for pos in list(range(0, len(aux))):
                    aux[pos]=self.get_genes(parent_A[gene][pos], parent_B[gene][pos], up, low)
                
                child.append(aux)
                
            elif type(fill) == np.ndarray:
                #If the gene is a numpy array
                   
                dims = fill.shape
                
                if len (dims) == 1:
                    t = type(fill[0])
                    
                elif len (dims) == 2:
                    t = type(fill[0,0])
                    
                else:
                    t = type(fill[0,0,0])
                
                del (fill)
                
                fill = np.zeros(dims).astype(t)
                
                if len(dims) == 1:
                    
                    for i in list(range(0, dims[0])):
                        
                        gen = self.get_genes(parent_A[gene][i], parent_B[gene][i], up, low)
                        fill[i] = gen
                        
                elif len(dims) == 2:
                    for i in list(range(0, dims[0])):
                        for j in list(range(0, dims[1])):
                            fill[i, j] = self.get_genes(parent_A[gene][i, j], parent_B[gene][i, j], up, low)
                else:
                    #Max: 3-dimension numpy array
                    for i in list(range(0, dims[0])):
                        for j in list(range(0, dims[1])):
                            for k in list(range(0, dims[2])):
                                fill[i, j, k] = self.get_genes(parent_A[gene][i, j, k], parent_B[gene][i, j, k], up, low)
                
                child.append(fill)
                
            else:
                #If the gene is an integer, a float or a boolean
                fill = self.get_genes(parent_A[gene], parent_B[gene], up, low)
                
                child.append(fill)
                
            del (fill)
            gene +=1
            
        #Apply the objective function to the child
        child.append(self.objective_function(child))  

        return copy.deepcopy(child)
      
    def pair_couples_routes(self, parent_A, parent_B):
        
        sol_parent_A = copy.copy(parent_A[0])
        sol_parent_B = copy.copy(parent_B[0])
        
        
        accepted_sol = False
        while_loop_counter = 0
        
        while not(accepted_sol):
            
            solution = list()
            
            #Starts at the initial point
            solution.append(copy.deepcopy(self.initial_route_point))
            
            for i in list(range(0, len(sol_parent_A) - 2)):
                
                possible_steps = list()
                index = 0
                
                index = sol_parent_A.index(solution[-1])
                possible_steps.extend(copy.deepcopy([sol_parent_A[index - 1], sol_parent_A[index + 1]]))
                
                index = 0
                index = sol_parent_B.index(solution[-1])
                possible_steps.extend(copy.deepcopy([sol_parent_B[index - 1], sol_parent_B[index + 1]]))
                
                step_found = False
                checked_values = 0
                
                while not(step_found):
                    chosen = self.random.choice(possible_steps)
                    
                    if not(chosen in solution):
                        solution.append(copy.deepcopy(chosen))
                        step_found = True
                    else:
                        possible_steps.remove(chosen)
                        checked_values += 1
                        
                    if checked_values == 4:
                        
                        all_destinies = set(self.destinies)
                        chosen_destinies = set(solution)
                        not_chosen = all_destinies.difference(chosen_destinies)
                        solution.append(copy.deepcopy(not_chosen.pop()))
                        step_found = True
                    
            solution.append(copy.deepcopy(self.initial_route_point))
            distance = self.objective_function(solution)
            
            if len(set(solution)) == len(self.destinies):
                if distance < math.inf:
                    accepted_sol = True
                    
                else:
                    while_loop_counter += 1
                    
            elif while_loop_counter > len(self.destinies)*0.2:
                print('Fallo a la hora de crear un hijo')
                x.shape
            
        
        return copy.deepcopy([solution, distance])
    
    def get_genes(self, gene_A, gene_B, up_lim, down_lim):
        """Function which mixes the genes of two parents, while taking into account
        the limits of the input variable, and returns the new gene"""
        
        if type(gene_A) == bool:
            if gene_A == gene_B:
                if self.random.random() <= self.mutation_probability:
                    return not(gene_A)
                else:
                    return (gene_A)
            else:
                return self.random.choice(True, False)
            
        elif type(gene_A) == int:
            new_gene = self.compare_binaries(gene_A, gene_B)
            
            while not(down_lim <= new_gene <= up_lim):
                new_gene = self.compare_binaries(gene_A, gene_B)
            
            return new_gene
        
        else:
            
            g_int_A = int(round(gene_A))
            g_int_B = int(round(gene_B))
            g_int_dec_A = int(round((gene_A-g_int_A)*10**self.float_precission))
            g_int_dec_B = int(round((gene_B-g_int_B)*10**self.float_precission))
            
            new_int = self.compare_binaries(g_int_A, g_int_B)
            new_dec = self.compare_binaries(g_int_dec_A, g_int_dec_B)
            new_gene = new_int + new_dec/(10**self.float_precission)
            
            while not(down_lim <= new_gene <= up_lim):
                new_int = self.compare_binaries(g_int_A, g_int_B)
                new_dec = self.compare_binaries(g_int_dec_A, g_int_dec_B)
                new_gene = round(new_int + new_dec/(10**self.float_precission), self.float_precission)
                
            return new_gene
            
        return
    
    def compare_binaries (self, gene_A, gene_B):
        """ Function which compares the binary shape of two genes (integers) and
        returns a new gene (integer)"""
        
        gene_A = bin(gene_A)
        gene_B = bin(gene_B)
        
        len_diff = abs(len(gene_A) - len(gene_B))
        
        if len(gene_A) > len(gene_B):
            gene_B = '0b' + '0' * len_diff + gene_B[2:]
        elif len(gene_A) < len(gene_B):
            gene_A = '0b' + '0' * len_diff + gene_A[2:]
            
        new_gene = '0b'
        
        for i in list(range(2,len(gene_A))):
            
            if gene_A[i] == gene_B[i]:
                
                if self.random.random() <= self.mutation_probability:
                    if gene_A[i] == '0':
                        new_gene = new_gene + '1'
                    else:
                        new_gene = new_gene + '0'
                else:
                     new_gene = new_gene + gene_A[i]
                
            else:
                new_gene = new_gene + self.random.choice(['0','1'])
        
        if (new_gene[2] == 'b'):
            new_gene = new_gene.replace('b', '', 1)
        
        return int(new_gene, 2)
    
    def sort_population(self):
        """ Function which sortens the population list depending on the score
        of the members at the objective function. From the best ones to the
        worst ones"""
        
        score = list()
        
        #calculates the scores of each member of the population
        for member in self.population:
            score.append(member[-1])
            
        #Gets the order from the lowest punctuation to the highest
        
#        print(score)
        order = list(np.argsort(score))
        
        score = score.clear()
        score = list()
        
        if self.objective == 'min':
            for pos in order:
                score.append(copy.deepcopy(self.population[pos]))
        else:
            for i in list(range(0, len(order))):
                score.append(copy.deepcopy(self.population[order[-1-i]]))
        
        self.population.clear() 
        
        self.population = score
        
        return
    
    def save_metadata(self, closing = False):
        
        if not(closing):
            self.metadata_file.write('Heuristic algorithm: '+ self.__class__.__name__ + ' \n')
            self.metadata_file.write('Objective: ' + self.objective + ', mixing type: ' + self.mix_type + '\n')
            
            actual_date = time.localtime()
            datetime_info =  'Executed the day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            
        else:
            actual_date = time.localtime()
            datetime_info =  'Stopped at day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            self.metadata_file.write('Final loss: ' + str(self.best_solution_found[-1]) + '\n')
            self.metadata_file.write('After ' + str(self.actual_iteration) + ' iterations.\n')
            self.metadata_file.write('Variable names: \n')
            for name in self.var_names:
                self.metadata_file.write('> ' + name + '\n')
                
            self.metadata_file.write('Parameters: \n')
            self.metadata_file.write('> Population amount: ' +  str(self.population_amount) + '\n')
            self.metadata_file.write('> Mutation probability: ' +  str(self.mutation_probability) + '\n')
            self.metadata_file.write('> Generations per internal iteration: ' +  str(self.generations_per_iteration) + '\n')
            self.metadata_file.write('> Percentage of best parents chosen in mix: ' +  str(self.best_parent_percentage) + '\n')
            self.metadata_file.write('> Percentage of worst parents chosen in mix: ' +  str(self.worst_parent_percentage) + '\n')
            
            
            
        
        return
    
class simulated_annealing (heuristic_algorithm):
    
    temperature_function = None
    temp_func_name = None
    get_next_sol = None
    
    actual_temperature = 0
    initial_temp = 0
    last_result = 0
    max_bad_iterations = 10
    
    next_temp = 0
    at_iteration = 0
    decline_constant = 0
    
    actual_solution = list()
    temperature_history = list()
    
    k = 1
    
    def __init__ (self):
        super().__init__()
        return
        
    def set_algorithm_values(self, temperature_function = 'default', decline_const = 0, initial_temp = 0, max_bad_iterations = 100):
        
        self.initial_temp = initial_temp
        self.actual_temperature = initial_temp
        self.max_bad_iterations = max_bad_iterations
        
        if type(temperature_function) == str:
            
            self.temp_func_name = temperature_function
            
            if temperature_function == 'default':
                self.temperature_function = self.exponential_temperature
                
            elif temperature_function == 'logaritmic':
                self.temperature_function = self.logaritmic_temperature
                
            elif temperature_function == 'linear':
                
                if decline_const == 0:
                    print(' >> ERROR: a decline constant is required to set the linear temperature function')
                    print(' >>> Default temperature function set. Exponential temperature.')
                    self.temperature_function = self.exponential_temperature
                else:
                    self.decline_constant = decline_const
                    self.temperature_function = self.linear_temperature
                    
            elif temperature_function == 'euler':
                
                if decline_const == 0:
                    print(' >> ERROR: a decline constant is required to set the Euler temperature function')
                    print(' >>> Default temperature function set. Exponential temperature.')
                    self.temperature_function = self.exponential_temperature
                else:
                    self.decline_constant = decline_const
                    self.temperature_function = self.euler_temperature
                    
            else:
                print(' >> ERROR: the specified temperature function is not programmed')
                print(' >>> Default temperature function set. Exponential temperature.')
                self.temperature_function = self.exponential_temperature
                
                
        else:
            self.temp_func_name = "user's temperature function"
            self.temperature_function = temperature_function
            
    def set_objective_func(self, h):
        
        if self.mix_type == 'parameters':
            self.get_next_sol = self.get_next_parameter_sol
        elif self.mix_type == 'traveling salesman':
            self.get_next_sol = self.get_next_route_sol
        else:
            print('>> ERROR: non identified mix type')
            print('>>> Please, choose between "parameters" or "traveling salesman"')
            
            return False
            
        self.objective_function = h
        
        return True
    
    def iterate(self, n_iterations, graph_process = False, graph_each_iterations = 10, initial_sol = None):
        
        if not(self.already_working):
            self.initial_time = time.time()
            self.already_working = True
        
        if initial_sol == None:
            #Get an initial solution
            sol = self.create_random_solution()
        else:
            sol = initial_sol
            
        self.best_solution_found = copy.deepcopy(sol)
        
        #Store the actual loss value
        self.last_result = sol[-1]
        
        self.initial_time = time.time()
        
        for i in list(range(0, n_iterations)):
            
            scape = False
            its_without_performance = 0
            
            #Update the temperature
            self.actual_temperature = self.temperature_function([i, n_iterations, self.initial_temp, self.actual_temperature, self.best_solution_found])
            
            while not(scape):
                
                new_sol = self.get_next_sol(sol)
                
                if self.objective == 'min':
                    #Case minimize
                    if (new_sol[-1]< self.last_result) or (self.check_if_accept(new_sol[-1])):
                        #Case the new solution is better than the old one, or it
                        # is worse but we accept the new solution
                        self.last_result = new_sol[-1]
                        sol.clear()
                        sol = list()
                        sol = copy.deepcopy(new_sol)
                        self.actual_solution = copy.deepcopy(new_sol)
                        
                        if new_sol[-1] < self.best_solution_found[-1]:
                            self.best_solution_found = list()
                            self.best_solution_found = copy.deepcopy(new_sol)
                            
                        del (new_sol)
                        scape = True
                        
                    else:
                        #Case the new solution is worst than the old one
                        #and we do not accept that solution
                        its_without_performance += 1
                        del (new_sol)
                    
                        
                else:
                    #Case maximize
                    if (new_sol[-1] > self.last_result) or (self.check_if_accept(new_sol[-1])):
                        #Case the new solution is better than the old one, or it
                        # is worse but we accept the new solution
                        self.last_result = new_sol[-1]
                        sol.clear()
                        sol = list()
                        sol = copy.deepcopy(new_sol)
                        
                        self.actual_solution = copy.deepcopy(sol)
                        
                        if new_sol[-1] > self.best_solution_found[-1]:
                            self.best_solution_found = list()
                            self.best_solution_found = copy.deepcopy(new_sol)
                        
                        del (new_sol)
                        scape = True
                        
                        
                    else:
                        #Case the new solution is worst than the old one
                        #and we do not accept that solution
                        
                        its_without_performance += 1
                        del (new_sol)
                        
                if graph_process:
                    if i%10 == 0:
                        self.graph_current_optimization_state()
                       
                if (its_without_performance >= self.max_bad_iterations):
                    print('Iteration stopped after ', self.max_bad_iterations, ' iterations without performance')
                    self.actual_solution = copy.deepcopy(sol)
                    return
                
            self.temperature_history.append(self.actual_temperature)
            self.iterations_history.append(self.actual_iteration)
            self.actual_iteration +=1
            self.time_history.append(time.time() - self.initial_time)
            self.losses_history.append(self.best_solution_found[-1])
            
            if self.save_info:
                self.save_optimization_info()
                
        return
        
    def get_next_parameter_sol(self, actual_sol):
        
        child = list()
            
        gene = 0
            
        for parameter in self.parameters:
            
            fill = copy.copy(parameter.get('initial_value'))
            
            up = parameter.get('upper_limit')
            low = parameter.get('lower_limit')
            
            if up == math.inf:
                up = abs(low * 100)
            elif low == -math.inf:
                low = - abs(up * 100)
            
            sigma = self.k * (up - low)/6
                                
            if type(fill) == list:
                #If the gene is a list of floats, integers or booleans
                aux = copy.deepcopy(actual_sol[gene])
                
                for pos in list(range(0, len(aux))):
                    a = self.return_random_gauss(actual_sol[gene][pos], sigma)
                    
                    if a < low:
                        a = low
                        
                    elif a> up:
                        a = up
                        
                    aux[pos] = a
                    
                child.append(copy.deepcopy(aux))
                
            elif type(fill) == np.ndarray:
                #If the gene is a numpy array
                   
                dims = fill.shape
                sigma =  sigma * np.ones(dims)
                
                a = np.random.normal(loc = actual_sol[gene], scale = sigma)
                
                if len (dims) == 1:
                    
                    for pos in list(range(0,dims[0])):
                        if a [pos] < low:
                            a[pos] = low
                        
                        if a [pos] > up:
                            a[pos] = up
                        
                elif len (dims) == 2:
                    
                    for i in list(range(0,dims[0])):
                        for j in list(range(0,dims[1])):
                            if a [i, j] < low:
                                a[i, j] = low
                            
                            if a [i, j] > up:
                                a[i, j] = up
                            
                else:
                    for i in list(range(0,dims[0])):
                        for j in list(range(0,dims[1])):
                            for k in list(range(0,dims[2])):
                                if a [i, j, k] < low:
                                    a[i, j, k] = low
                                
                                if a [i, j, k] > up:
                                    a[i, j, k] = up
                                
                child.append(a)
                
            elif type(fill) == bool:
                #If the gene is a boolean
                child.append(self.random.choice(True, False))
                
            else:
                #If the gene is an integer or a float
                a = self.return_random_gauss(mu = actual_sol[gene], sigma = sigma)

                if a < low:
                    a = low
                    
                elif a> up:
                    a = up
                    
                child.append(a)
                
            del (fill)
            
            gene +=1
                
        #Apply the objective function to the child
        child.append(self.objective_function(child))  
        
        return copy.deepcopy(child)
    
    def get_next_route_sol(self, actual_sol):
        
        new_sol = list()
        
        actual_route = actual_sol[0]
        
        accept_route = False
        
        while(not(accept_route)):
            
            new_route = list()
            
            possible_twists = list(range(1, len(actual_route) - 1))
            
            initial_position_to_twist = self.random.choice(possible_twists)
            
            possible_twists = list(range(initial_position_to_twist + 1, len(actual_route)))
            
            if len(possible_twists)> 1:
                final_position_to_twist = self.random.choice(possible_twists)
            else:
                final_position_to_twist = possible_twists[0]
            
            new_route = copy.deepcopy(actual_route[0:initial_position_to_twist])
            
            twisting = copy.deepcopy(actual_route[initial_position_to_twist:final_position_to_twist])
            twisting.reverse()
            new_route.extend(twisting)
            new_route.extend(copy.deepcopy(actual_route[final_position_to_twist:]))
            
            distance = self.objective_function(new_route)
            
            if not(distance == math.inf):
                accept_route = True
                new_sol = [new_route, distance]
            
            if (len(set(new_route)) != len(self.destinies)) or ((len(new_route)) != len(self.destinies) + 1):
                print('problema de repetición de ciudades')
                x.shape
                
        return copy.deepcopy(new_sol)
    
    def check_if_accept(self, actual_result):
        
        if self.actual_temperature>0:
            
            if self.objective == 'min':
                x = (actual_result - self.last_result)/self.actual_temperature
            else:
                x = -(actual_result - self.last_result)/self.actual_temperature
                
            accept_probab = math.exp(-x)
            
            if self.random.random() <= accept_probab:
                return True
            else:
                return False
        else:
            return False
        
    def save_metadata(self, closing = False):
        
        if not(closing):
            self.metadata_file.write('Heuristic algorithm: '+ self.__class__.__name__ + ' \n')
            self.metadata_file.write('Objective: ' + self.objective + ', mixing type: ' + self.mix_type + '\n')
            
            actual_date = time.localtime()
            datetime_info =  'Executed the day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            
        else:
            actual_date = time.localtime()
            datetime_info =  'Stopped at day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            self.metadata_file.write('Final loss: ' + str(self.best_solution_found[-1]) + '\n')
            self.metadata_file.write('After ' + str(self.actual_iteration) + ' iterations.\n')
            self.metadata_file.write('Initial temperature: ' +  str(self.initial_temp) + '\n')
            self.metadata_file.write('Final temperature: ' +  str(self.actual_temperature) + '\n')
            self.metadata_file.write('Variable names: \n')
            for name in self.var_names:
                self.metadata_file.write('> ' + name + '\n')
                
            self.metadata_file.write('Parameters: \n')
            
            self.metadata_file.write('> Temperature function: ' +  self.temp_func_name + '\n')
            
            self.metadata_file.write('> Initial temperature: ' +  str(self.initial_temp) + '\n')
        
        return
    
    def save_optimization_info(self):
        
        self.info_writer. writerow([str(self.iterations_history[-1]), str(round(1000*(time.time()-self.initial_time))), str(self.losses_history[-1]), str(self.temperature_history[-1])])
        
        return
    
    def graph_current_optimization_state(self):
        
        if self.losses_figure == None:
            self.losses_figure, self.axs = plt.pyplot.subplots(1, 2)
        
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[0].plot(self.iterations_history, self.losses_history)
        self.axs[1].plot(self.iterations_history, self.temperature_history)
        
        plt.pyplot.pause(0.05)
        
        return
    
    def exponential_temperature(self, args):
        
        it = args[0]
        
        new_temp = self.initial_temp*0.95**it
            
        return new_temp
    
    def logaritmic_temperature(self, args):
        
        actual_iteration = args[0]
        
        if actual_iteration>1:
            new_temp = self.initial_temp / math.log(actual_iteration)
        else:
            new_temp = self.initial_temp
        
        return new_temp
        
    def linear_temperature(self, args):
        
        new_temp = self.decline_constant * self.actual_temperature
        
        return new_temp
    
    def euler_temperature(self, args):
        
        actual_iteration = args[0]
        
        new_temp = math.exp(-self.decline_constant * actual_iteration)* self.initial_temp
        
        return new_temp
    
class genetic_adaptative_algorithm(genetic_algorithm, simulated_annealing, heuristic_algorithm):
    
    min_evol_years = 16
    max_evol_years = 42
    
    def __init__ (self):
        super().__init__()
        self.initial_temp = 0
        self.actual_temperature = 0
        self.max_bad_iterations = 365
        
        self.temp_func_name = 'default'
        self.temperature_history = [0]
        self.temperature_function = self.exponential_temperature
        
        return
    
    def set_algorithm_values(self, pop_amount = 10, generations_per_iteration = 10, max_adapt_years = 42, min_adapt_years = 16, search_percentage = 2.5, mutation_probab = 0.1, best_parent_selection = 0.4, worse_parent_selection = 0.2):
        
        self.population_amount = pop_amount
        self.generations_per_iteration = generations_per_iteration
        self.max_evol_years = max_adapt_years
        self.min_evol_years = min_adapt_years
        self.k = search_percentage / 100
        self.mutation_probability = mutation_probab
        self.best_parent_percentage = best_parent_selection
        self.worst_parent_percentage = worse_parent_selection
        
        return
    
    def set_objective_func(self, h):
        
        if self.mix_type == 'parameters':
            self.pair_couple = self.pair_copules_parameters
            self.get_next_sol = self.get_next_parameter_sol
            
        elif self.mix_type == 'traveling salesman':
            self.pair_couple = self.pair_couples_routes
            self.get_next_sol = self.get_next_route_sol
            
        else:
            print('>> ERROR: non identified mix type')
            print('>>> Please, choose between "parameters" or "traveling salesman"')
            
            return False
            
        self.objective_function = h
        
        return True
    
    def get_initial_population(self):
        
        self.population = list()
        
        for i in range(0, self.population_amount):
            self.population.append(self.create_random_solution())
        
        self.evolve_population()
        self.sort_population()
        
        return
    
    def iterate(self):
        
        if not(self.already_working):
            self.initial_time = time.time()
            self.already_working = True
            
        for i in range(0, self.generations_per_iteration):
            self.mix_population()
            self.evolve_population()
            
            self.time_history.append(time.time() - self.initial_time)
            self.actual_iteration += 1
            self.iterations_history.append(self.actual_iteration)
            self.losses_history.append(self.population[0][-1])
            
            if self.save_info:
                self.save_optimization_info()
        
        self.sort_population()
        self.best_solution_found = self.population[0]
        
        return
    
    def evolve_population(self):
        
        evolved_population = list()
        
        for member in self.population:
            evolved_population.append(self.iterate_s_ann(n_iterations = int(self.random.uniform(low = self.min_evol_years, up = self.max_evol_years)), sol = member))
        
        
        self.population = list()
        self.population = copy.deepcopy(evolved_population)
        del (evolved_population)
        
        return
    
    def iterate_s_ann(self, n_iterations, sol):
            
        best_solution_found = copy.deepcopy(sol)
        
        #Store the actual loss value
        last_result = sol[-1]
        
        for i in list(range(0, n_iterations)):
            
            scape = False
            its_without_performance = 0
            
            while not(scape):
                
                new_sol = self.get_next_sol(sol)
                
                if self.objective == 'min':
                    #Case minimize
                    if (new_sol[-1]< last_result):
                        #Case the new solution is better than the old one, or it
                        # is worse but we accept the new solution
                        last_result = new_sol[-1]
                        sol = list()
                        sol = copy.deepcopy(new_sol)
                        
                        if new_sol[-1] < best_solution_found[-1]:
                            best_solution_found = list()
                            best_solution_found = copy.deepcopy(new_sol)
                            
                        del (new_sol)
                        scape = True
                        
                    else:
                        #Case the new solution is worst than the old one
                        #and we do not accept that solution
                        its_without_performance += 1
                        del (new_sol)
                    
                        
                else:
                    #Case maximize
                    if (new_sol[-1] > last_result):
                        #Case the new solution is better than the old one, or it
                        # is worse but we accept the new solution
                        last_result = new_sol[-1]
                        sol = list()
                        sol = copy.deepcopy(new_sol)
                        
                        if new_sol[-1] > best_solution_found[-1]:
                            best_solution_found = list()
                            best_solution_found = copy.deepcopy(new_sol)
                        
                        del (new_sol)
                        scape = True
                        
                    else:
                        #Case the new solution is worst than the old one
                        #and we do not accept that solution
                        
                        its_without_performance += 1
                        del (new_sol)
                        
                if (its_without_performance >= self.max_bad_iterations):
                    
                    return copy.deepcopy(best_solution_found)
                
        return copy.deepcopy(best_solution_found)
    
    def graph_current_optimization_state(self):
        
        if self.losses_figure == None:
            self.losses_figure = plt.pyplot.figure()
            self.axs = self.losses_figure.gca()
        
        
        self.axs.cla()
        self.axs.plot(self.iterations_history, self.losses_history)
        plt.pyplot.pause(0.05)
        
        return
    
    def save_metadata(self, closing = False):
        
        if not(closing):
            self.metadata_file.write('Heuristic algorithm: '+ self.__class__.__name__ + ' \n')
            self.metadata_file.write('Objective: ' + self.objective + ', mixing type: ' + self.mix_type + '\n')
            
            actual_date = time.localtime()
            datetime_info =  'Executed the day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            
        else:
            actual_date = time.localtime()
            datetime_info =  'Stopped at day ' + str(actual_date.tm_mday) + ' month ' + str(actual_date.tm_mon) + ' hour ' + str(actual_date.tm_hour)+ ':' + str(actual_date.tm_min)+ ':' + str(actual_date.tm_sec)
            self.metadata_file.write(datetime_info + '\n')
            self.metadata_file.write('Final loss: ' + str(self.best_solution_found[-1]) + '\n')
            self.metadata_file.write('After ' + str(self.actual_iteration) + ' iterations.\n')
            self.metadata_file.write('Variable names: \n')
            for name in self.var_names:
                self.metadata_file.write('> ' + name + '\n')
                
            self.metadata_file.write('Parameters: \n')
            self.metadata_file.write('> Population amount: ' +  str(self.population_amount) + '\n')
            self.metadata_file.write('> Mutation probability: ' +  str(self.mutation_probability) + '\n')
            self.metadata_file.write('> Minimum years of adaptation: ' +  str(self.min_evol_years) + '\n')
            self.metadata_file.write('> Maximum years of adaptation: ' +  str(self.max_evol_years) + '\n')
            self.metadata_file.write('> Generations per internal iteration: ' +  str(self.generations_per_iteration) + '\n')
            self.metadata_file.write('> Percentage of best parents chosen in mix: ' +  str(self.best_parent_percentage) + '\n')
            self.metadata_file.write('> Percentage of worst parents chosen in mix: ' +  str(self.worst_parent_percentage) + '\n')
            
        return