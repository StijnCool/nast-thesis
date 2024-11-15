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
        n_p = collocation_point_numbers # u-, u+, v-, v+
        B_p = collocation_point_bounds # u-, u+, v-, v+
        fp = fixed_point # u, v

        um = np.linspace(fp[0]-B_p[0], fp[0], n_p[0]+1) # u- to u*
        up = np.linspace(fp[0], fp[0]+B_p[1], n_p[1]+1) # u* to u+
        vm = np.linspace(fp[1]-B_p[2], fp[1], n_p[2]+1) # v- to v*
        vp = np.linspace(fp[1], fp[1]+B_p[3], n_p[3]+1) # v* to v+

        u_mp, v_mp = np.meshgrid(um, vp, indexing='ij') # exclude none
        u_pp, v_pp = np.meshgrid(up[1:], vp, indexing='ij') # exclude u*
        u_mm, v_mm = np.meshgrid(um, vm[:-1], indexing='ij') # exclude v*
        u_pm, v_pm = np.meshgrid(up[1:], vm[:-1], indexing='ij') # exclude u* and v*

        u_vp = np.concatenate((u_mp, u_pp))
        v_vp = np.concatenate((v_mp, v_pp))
        u_vm = np.concatenate((u_mm, u_pm))
        v_vm = np.concatenate((v_mm, v_pm))

        u = np.concatenate((u_vm, u_vp), axis=1)
        v = np.concatenate((v_vm, v_vp), axis=1)
        
        grid = np.array([u,v])

        points = np.vstack([u.ravel(order='F'), v.ravel(order='F')])

        return points, grid
    
    
    def calculate_smoothness(self, collocation_point_numbers, collocation_point_bounds):
        return np.min(np.divide(collocation_point_bounds, collocation_point_numbers))
    
    
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
                
    