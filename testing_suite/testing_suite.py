import numpy as np
import copy

class Suite:
    
    def __init__(self, theory_parameters, meta_parameters, control_parameters, export_directory, debug=True):
        self.theory_parameters = theory_parameters # code, beta_functions, fixed_point, true_result, linear_generating_function, M
        self.meta_parameters = meta_parameters # code, collocation_point_numbers, collocation_point_bounds, basis_functions, basis_functions_derivatives, fix_parameters, population_size, algorithm
        self.control_parameters = control_parameters # code, fitness_function, selection, mutation, crossover, iteration_cap
        self.export_directory = export_directory
        self.N = self.theory_parameters.fixed_point.size()
        self.Tm = self.meta_parameters.get_combination_number()
        self.Tc = self.control_parameters.get_combination_number()
        self.Tt = self.Tm * self.Tc
        self.Cc = 1
        self.Cm = 1
        self.Ct = 1
        
    def run_all(self):
        for meta_parameter_set in self.meta_parameters.get_all_combinations():
            collocation_points, grid = generate_grid()
            N_p = collocation_points.size()
            smoothness = calculate_smoothness()
            F, dF, phi, dphi, beta = precalculate_tensors()
            if meta_parameter_set.algorithm.has_population:
                population = generate_starting_population(meta_parameter_set.population_size, N_p)
            self.Cc = 1
            for control_parameter_set in self.control_parameters.get_all_combinations():
                debug(True, 1)
                algorithm = copy.deepcopy(meta_parameter_set.algorithm)
                algorithm.set_control_parameters(control_parameter_set)
                algorithm.run()
                algorithm.save_result(self.export_directory)
                self.Cc += 1
                self.Ct += 1
            self.Cm += 1
            
    def run_batch(self, batch_sizes):
        return None
            
    def generate_grid(self, fixed_point, collocation_point_numbers, collocation_point_bounds):
        return None, None
    
    def calculate_smoothness(self):
        return None
    
    def precalculate_tensors(self):
        return None, None, None, None, None
    
    def generate_starting_population(self, population_size, N_p, fix_parameters):
        population = np.random.rand(population_size, N_p, self.M)*2-1
        if fix_parameters:
            return np.array([fix_parameters(individual) for individual in population])
        else:
            return population
        
    def fix_parameters(self):
        return None
    
    def debug(boolean, number):
        if boolean:
            if number == 1:
                print(f'Meta: {self.Cm}/{self.Tm} ({round(self.Cm/self.Tm*100,2)}%%); Control: {self.Cc}/{self.Tc} ({round(self.Cc/self.Tc*100,2)}%%); Total: {self.Ct}/{self.Tt} ({round(self.Ct/self.Tt*100,2)}%%);', end='\r')
            elif number == 2:
                print('2')
            else:
                raise Exception('Unknown debug request.')
                
    