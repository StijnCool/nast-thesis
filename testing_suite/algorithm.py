import numpy as np
import math
import random as rand
from timeit import default_timer as timer
import copy

class Algorithm:
    
    
    def __init__(self, code, has_population, **kwargs):
        self.code = code
        self.has_population = has_population
        self.__dict__.update(kwargs)
        
        
    def set_control_parameters(self, control_parameter_set, pc_beta, pc_dpsi):
        self.__dict__.update(control_parameter_set.__dict__)
        
        
    def run(self):
        raise NotImplementedError('Subclasses should implement.')
        
    
    def save_result(self, directory):
        raise NotImplementedError('Subclasses should implement.')
    
    
    def generate_filename(self):
        raise NotImplementedError('Subclasses should implement.')


# --------------------------------------------------------------------------------------------------------------------------

        
class GeneticAlgorithm(Algorithm):
    # code, has_population, population, keep_percent, crossover_percent, mutation_percent, mutation_amount, fitness_code, selection_code, mutation_code, crossover_code, iteration_cap
    
    
    def __init__(self, code, has_population, **kwargs):
        self.code = code
        self.has_population = has_population
        self.__dict__.update(kwargs)
        self.fitness_dict = {'abs': self.f_abs} # fitness takes self, pop; returns scores
        self.selection_dict = {'roulette': self.s_roulette} # selection takes self, pop, scores; returns pop, scores
        self.mutation_dict = {'gaussian': self.m_gaussian} # mutation takes self, pop, scores; returns pop
        self.crossover_dict = {'one_point': self.c_one_point} # crossover takes self, q1, q2; returns q1, q2
    
    
    def set_control_parameters(self, control_parameter_set, population, beta, dpsi):
        self.__dict__.update(control_parameter_set.__dict__)
        self.pop = population
        self.beta = beta
        self.dpsi = dpsi
        self.pop_size = self.pop.shape[0]
        self.fitness = self.fitness_dict[self.fitness_code]
        self.selection = self.selection_dict[self.selection_code]
        self.mutation = self.mutation_dict[self.mutation_code]
        self.crossover = self.crossover_dict[self.crossover_code]
        self.keep_number = int(self.pop_size*self.keep_percent/2)*2
        
    
    def run(self):
        pop = copy.deepcopy(self.pop)
        self.result = {
            'best_score': []
        }
        counter = 1
        scores = self.fitness(pop)
        start = np.zeros(6)
        end = np.zeros(6)
        start[0] = timer()
        while counter <= self.iteration_cap:
            start[1] = timer()
            pop, scores = self.selection(pop, scores)
            end[1] = timer()
            start[2] = timer()
            pop = self.mutation(pop, scores)
            end[2] = timer()
            start[3] = timer()
            pop = self.shuffle_pop(pop)
            end[3] = timer()
            start[4] = timer()
            for i in range(0,self.pop_size,2):
                pop[i], pop[i+1] = self.crossover(pop[i], pop[i+1])
            end[4] = timer()
            start[5] = timer()
            scores = self.fitness(pop)
            end[5] = timer()
            self.result['best_score'].append(np.min(scores))
            counter += 1
        end[0] = timer()
        print(f'dt[t,s,m,sh,c,f] = {end-start}')
        return self.result['best_score']
    
    
    def shuffle_pop(self, pop):
        idx = np.random.rand(self.pop_size).argsort(axis=0)
        return np.take_along_axis(pop, np.tile(idx, (*pop.shape[:-1],1)).transpose(), axis=0)
        

    def f_abs(self, pop):
        return np.array([min(10.0**100, np.sum(np.abs(np.trace(np.matmul(self.beta, np.transpose(np.matmul(self.dpsi, p), axes=[2,0,1])), axis1=1, axis2=2)))) for p in pop], dtype=np.float32)
    
        
    def s_roulette(self, pop, scores):
        weights = np.max(np.abs(scores)) - np.abs(scores)
        pop = pop[scores.argsort()[::-1]] # sorts in descending order of scores i.e. best is at bottom of pop
        scores = scores[scores.argsort()[::-1]]
        order = np.arange(0, self.pop_size, 1)
        replacement_mask = None
        if self.keep_number > 0:
            replacement_mask = np.concatenate((np.array(rand.choices(order[self.keep_number+1:], weights=weights[self.keep_number+1:], k=self.pop_size-self.keep_number)), order[-self.keep_number:]))
        else:
            replacement_mask = np.array(rand.choices(order[self.keep_number+1:], weights=weights[self.keep_number+1:], k=self.pop_size-self.keep_number))
        pop = np.take_along_axis(pop, np.tile(replacement_mask, (*pop.shape[:-1],1)).transpose(), axis=0)
        scores = np.take_along_axis(scores, replacement_mask, axis=0)
        return pop, scores
        
        
    def m_gaussian(self, pop, scores):
        gaussian_values = np.random.normal(np.zeros(self.pop_size), self.mutation_amount*scores, pop.transpose().shape)
        probability_mask = np.random.binomial(1, self.mutation_percent, size=pop.transpose().shape)
        return np.add(pop, np.multiply(gaussian_values, probability_mask).transpose())
    
    
    def c_one_point(self, q1, q2):
        if rand.random() < self.crossover_percent:
            o_shape = q1.shape
            q1 = q1.reshape(o_shape[0]*o_shape[1])
            q2 = q2.reshape(o_shape[0]*o_shape[1])
            x = np.random.randint(1, o_shape[0]*o_shape[1]) # find a point along the population
            tmp = q2[:x].copy()
            q2[:x], q1[:x]  = q1[:x], tmp # 1-point crossover
            q1 = q1.reshape(*o_shape)
            q2 = q2.reshape(*o_shape)
        return q1, q2