import numpy as np

class Algorithm:
    
    def __init__(self, code, has_population, **kwargs):
        self.code = code
        self.has_population = has_population
        self.__dict__.update(kwargs)
        
    def set_control_parameters(self, control_parameter_set):
        self.__dict__.update(control_parameter_set.__dict__)
        
    def run(self):
        return None
    
    def save_result(self, directory):
        return None
    
    def generate_filename(self):
        return 'filename'