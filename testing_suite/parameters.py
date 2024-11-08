import numpy as np

class Parameters:
    
    def __init__(self, code, **kwargs):
        self.code = code
        self.__dict__.update(kwargs)
    
    def get_option_counts(self):
        return {key: np.array(self.__dict__[key]).size for key in self.__dict__.keys()-{'code'}}
    
    def get_combination_number(self):
        return np.prod(np.array([np.array(self.__dict__[key]).size for key in self.__dict__.keys()-{'code'}]))