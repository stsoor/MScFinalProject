import numpy as np

class RandomGenerator:
    def __init__(self, mode, param_1, param_2):
        '''
        @param mode: either 'uniform', or 'normal'
        @param param_1, param_2: either lo, hi or mean, and variance
        '''
        self.param_1 = param_1
        self.param_2 = param_2
        if mode == 'uniform':
            self.generator =  np.random.uniform
        if mode == 'normal':
            self.generator = np.random.normal
        else:
            raise ValueError('mode has to be "uniform" or "normal"')
 
    def __call__(self, size=None):
        return self.generator(self.param_1, self.param_2, size=size)
