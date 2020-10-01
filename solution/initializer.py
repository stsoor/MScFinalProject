import numpy as np

class Initializer:
    def __new__(self, problem, dimensionality):
        pass

class DistanceSolutionInitializer(Initializer):
    def __new__(self, problem, dimensionality):
        node_num = problem.hypergraph.shape[0]
        lower_bounds = problem.get_vector_lower_bounds()
        upper_bounds = problem.get_vector_upper_bounds()
        bounds_range = upper_bounds - lower_bounds

        x = np.random.uniform(0, 1, size=(dimensionality, node_num*3)) # current particle positions
        x = lower_bounds + x * bounds_range

        return x