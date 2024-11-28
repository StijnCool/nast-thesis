import numpy as np
import copy
import time

class Suite:
    
    
    def __init__(self, theory_parameters, meta_parameters, control_parameters, export_directory, do_debug=True):
        self.theory_parameters = theory_parameters # code, beta_functions, fixed_point, true_result, linear_generating_function, M
        self.meta_parameters = meta_parameters # code, collocation_point_numbers, collocation_point_bounds, basis_functions, basis_functions_derivatives, do_fix_parameters, population_size, algorithm
        self.control_parameters = control_parameters # code, fitness_function, selection, mutation, crossover, iteration_cap
        self.export_directory = export_directory
        self.do_debug = do_debug
        self.N = self.theory_parameters.fixed_point.size
        self.Tm = self.meta_parameters.get_combination_number()
        self.Tc = self.control_parameters.get_combination_number()
        self.Tt = self.Tm * self.Tc
        self.Cc = 1
        self.Cm = 1
        self.Ct = 1
        self.temp = []
        
        
    def run_all(self):
        for meta_parameter_set in self.meta_parameters.get_all_combinations():
            collocation_points, grid, core_ids = self.generate_grid(
                self.theory_parameters.fixed_point, 
                meta_parameter_set.collocation_point_numbers, 
                meta_parameter_set.collocation_point_bounds)
            
            N_p = collocation_points.shape[0]
            
            smoothness = self.calculate_smoothness(
                meta_parameter_set.collocation_point_numbers, 
                meta_parameter_set.collocation_point_bounds)
            
            F_fp, psi, dpsi, beta = self.precalculate_tensors(
                collocation_points, 
                self.theory_parameters.linear_generating_function, 
                meta_parameter_set.basis_functions, 
                meta_parameter_set.basis_functions_derivatives, 
                self.theory_parameters.beta_functions,
                N_p, 
                smoothness)
            
            population = None
            if meta_parameter_set.algorithm.has_population:
                population = self.generate_starting_population(
                    meta_parameter_set.population_size, 
                    N_p, 
                    meta_parameter_set.do_fix_parameters, 
                    core_ids, 
                    psi, 
                    F_fp)
                
            self.Cc = 1
            
            for control_parameter_set in self.control_parameters.get_all_combinations():
                self.debug(False, 2)
                self.debug(True, 1)
                time.sleep(1)
                
                algorithm = copy.deepcopy(meta_parameter_set.algorithm)
                algorithm.set_control_parameters(control_parameter_set, population, beta, dpsi)
                self.temp.append(algorithm.run())
                # algorithm.save_result(self.export_directory)
                
                self.Cc += 1
                self.Ct += 1
                
            self.Cm += 1
        print('')
            
            
    def run_batch(self, batch_sizes):
        return (np.array(self.temp))
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

        points = np.transpose(np.vstack([u.ravel(order='F'), v.ravel(order='F')]))

        return points, grid, np.array([])
    
    
    def calculate_smoothness(self, collocation_point_numbers, collocation_point_bounds):
        return np.min(np.divide(collocation_point_bounds, collocation_point_numbers))
    
    
    def precalculate_tensors(self, collocation_points, F_fp, psi, dpsi, beta, N_p, sigma):
        dpsi(collocation_points, N_p, sigma, self.N, np.transpose(collocation_points), collocation_points)
        return np.array([F_fp(cp) for cp in collocation_points]), psi(collocation_points, N_p, sigma, self.N, np.transpose(collocation_points)), dpsi(collocation_points, N_p, sigma, self.N, np.transpose(collocation_points), collocation_points), beta(collocation_points)
    
    
    def generate_starting_population(self, population_size, N_p, do_fix_parameters, core_ids, psi, F_fp):
        population = np.random.rand(population_size, N_p, self.theory_parameters.M)*2-1
        if do_fix_parameters:
            return np.array([self.fix_parameters(individual, core_ids, psi, F_fp) for individual in population])
        else:
            return population
        
        
    def fix_parameters(self, parameters, core_ids, psi, F_fp):
        pc = np.copy(parameters)
        for k in range(0,core_ids.size):
            kd = core_ids[k]
            pc[k,:] = 1/psi[kd,k] * np.subtract(F_fp[kd], np.add(np.matmul(psi[kd,0:k], pc[0:k,:]), np.matmul(psi[kd,k+1:], pc[k+1:,:])))
        return pc # p[ith basis function, function mu]
    
    
    def debug(self, boolean, number):
        if boolean:
            if number == 1:
                print(f'Meta: {self.Cm-1}/{self.Tm} ({round((self.Cm-1)/self.Tm*100,2)}%%); Control: {self.Cc-1}/{self.Tc} ({round((self.Cc-1)/self.Tc*100,2)}%%); Total: {self.Ct-1}/{self.Tt} ({round((self.Ct-1)/self.Tt*100,2)}%%);', end='\r')
            elif number == 2:
                print(f'{list(meta_parameter_set.__dict__.values())} and {list(control_parameter_set.__dict__.values())}')
            else:
                raise Exception('Unknown debug request.')
                
    
    def get_combination_number(self):
        return self.meta_parameters.get_combination_number() * self.control_parameters.get_combination_number()
                
    