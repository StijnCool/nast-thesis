import numpy as np
import itertools
import random as rand

class Parameters:
    
    
    def __init__(self, code, **kwargs):
        self.code = code
        self.__dict__.update(kwargs)
    
    
    def get_option_counts(self):
        return {key: np.array(self.__dict__[key]).shape[0] for key in self.__dict__.keys()-{'code'}}
    
    
    def get_combination_number(self):
        return np.prod(np.array([np.array(self.__dict__[key]).shape[0] for key in self.__dict__.keys()-{'code'}]))
    
    
    def get_all_combinations(self):
        permutations = np.array([])
        uncoded_dict = {key: self.__dict__[key] for key in self.__dict__.keys()-{'code'}}
        keys, values = zip(*uncoded_dict.items())
        for sub_dict in [dict(zip(keys, v)) for v in itertools.product(*values)]:
            sub_set = sub_dict 
            sub_set['parameter_code'] = self.code
            permutations = np.append(permutations, [ParameterSet(sub_dict)])
        return permutations

    
# --------------------------------------------------------------------------------------------------------------------------
    
    
class ParameterSet:
    
    
    def __init__(self, content):
        self.__dict__.update(content)
        
      
    def __str__(self):
        printer = "P"
        for k, v in self.__dict__.items():
            printer = printer + str(v)[3:5]
        return printer
    