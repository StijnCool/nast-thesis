import numpy as np
import math

class Algorithm:
    
    
    def __init__(self, code, has_population, **kwargs):
        self.code = code
        self.has_population = has_population
        self.__dict__.update(kwargs)
        
        
    def set_control_parameters(self, control_parameter_set):
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
        self.fitness_dict = {}
        self.selection_dict = {'roulette': self.s_roulette}
        self.mutation_dict = {'gaussian': self.m_gaussian}
        self.crossover_dict = {}
    
    
    def set_control_parameters(self, control_parameter_set, population):
        self.__dict__.update(control_parameter_set.__dict__)
        self.pop = population
        self.pop_size = self.pop.shape[0]
        self.fitness = self.fitness_dict[self.fitness_code]
        self.selection = self.selection_dict[self.selection_code]
        self.mutation = self.mutation_dict[self.mutation_code]
        self.crossover = self.crossover_dict[self.crossover_code]
        self.keep_number = int(self.pop_size*self.keep_percent/2)*2
        
    
    def run(self):
        self.result = {}
        
        
    def s_roulette(self):
        weights = np.max(np.abs(self.scores)) - np.abs(self.scores)
        self.pop = self.pop[self.scores.argsort()[::-1]] # sorts in descending order of scores i.e. best is at bottom of pop
        self.scores = self.scores[self.scores.argsort()[::-1]]
        self.pop[:-self.keep_number] = np.array(rand.choices(self.pop[self.keep_number:], weights=weights[keep_number:], k=pop_size-keep_number))
        
        
    def m_gaussian(self):
        gaussian_values = np.random.normal(np.zeros(self.scores.shape), self.mutation_amount*self.scores, self.pop.transpose().shape)
        probability_mask = np.random.choice([0, 1], size=self.pop.transpose().shape, p=[1-self.mutation_percent, self.mutation_percent])
        self.pop = np.add(self.pop, np.multiply(gaussian_values, probability_mask).transpose())
        return self.pop