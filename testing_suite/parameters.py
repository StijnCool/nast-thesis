import numpy as np

class Parameters:
    
    def __init__(self, code, **kwargs):
        self.code = code
        self.__dict__.update(kwargs)
    
    def get_option_counts(self):
        return {key: np.array(self.__dict__[key]).size for key in self.__dict__.keys()-{'code'}}
    
    def get_combination_number(self):
        return np.prod(np.array([np.array(self.__dict__[key]).size for key in self.__dict__.keys()-{'code'}]))
    
    def get_all_combinations(self):
        subset = {'code': self.code}
        for key in self.__dict__.keys()-{'code'}:
            if isinstance(self.__dict__[key], np.ndarray):
                subset[key] = self.__dict__[key][0]
            else:
                subset[key] = self.__dict__[key]
        return np.array([ParameterSet(subset)])
    
    
class ParameterSet:
    def __init__(self, content):
        self.__dict__.update(content)
    