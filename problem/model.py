import numpy as np

class Hypergraph:
    def __new__(self, node_num, edge_num):
        return np.full((node_num, edge_num), False, dtype=np.bool)
    
    def __getitem__(self, key): # to hide linter errors
        pass
    def __setitem__(self, key, value): # to hide linter errors
        pass

class RandomHypergraph:
    def __new__(self, node_num, edge_num, random_generator, inclusion_threshold):
        return random_generator((node_num, edge_num)) <= inclusion_threshold

class ProblemModel:
    def __init__(self, hypergraph, min_node_distance, canvas_width, canvas_height):
        self.hypergraph = hypergraph
        assert(min_node_distance > 0)
        self.min_node_distance = min_node_distance
        self.size = (canvas_width, canvas_height)
