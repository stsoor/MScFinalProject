import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix

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

class NodewiseDistanceModel(ProblemModel):
    def __init__(self, hypergraph, min_node_distance, canvas_width, canvas_height):
        super(NodewiseDistanceModel, self).__init__(hypergraph, min_node_distance, canvas_width, canvas_height)
        self._node_num, self._edge_num = self.hypergraph.shape

    def calculate_distance_matrix(self, x):
        positions = self.extract_positions2D(x)
        return distance_matrix(positions, positions, threshold=10**8) # pdist would give a more compact distance descriptor

    def _get_distance_thresholds(self, x):
        distance_thresholds = x[self._node_num*2:]
        return distance_thresholds

    def _calculate_min_thresholds(self, x):
        distance_thresholds = self._get_distance_thresholds(x)
        return np.minimum.outer(distance_thresholds, distance_thresholds) # (for np.minimum.outer) pylint: disable=no-member

    def _is_in_threshold(self, x):
        return self.calculate_distance_matrix(x) <= self._calculate_min_thresholds(x)

    def extract_positions2D(self, x):
        assert(len(x) == self._node_num*3)
        return x[:self._node_num*2].reshape((self._node_num, 2), order='F')

    def get_edge_components(self, x):
        assert(len(x) == self._node_num*3)
        all_components = []
        in_threshold_matrix = self._is_in_threshold(x)
        for edge_id in range(self._edge_num):
            edge = self.hypergraph[:, edge_id]
            edge_nodes = np.where(edge)[0]
            edge_non_diagonal_mask = ~np.diag(np.full(edge_nodes.size, True, dtype=np.bool))
            edge_in_threshold_matrix = in_threshold_matrix[edge_nodes,:][:,edge_nodes]
            edge_adjacency = edge_non_diagonal_mask & edge_in_threshold_matrix
            labels = edge_nodes
            edge_pd_adjacency = pd.DataFrame(edge_adjacency, index=labels, columns=labels)
            edge_graph = nx.from_pandas_adjacency(edge_pd_adjacency)
            edge_components = list(nx.connected_components(edge_graph))
            if len(edge_components) > 0:
                all_components.append(edge_components)
        
        return all_components
    
    def get_hypergraph_as_graph(self):
        incidence = (self.hypergraph @ self.hypergraph.T)
        np.fill_diagonal(incidence, False)
        return nx.from_numpy_matrix(incidence)

    def get_vector_lower_bounds(self):
        return np.full(self._node_num*3, 0.0, dtype=np.float32)

    def get_vector_upper_bounds(self):
        upper_bounds = np.empty(self._node_num*3, dtype=np.float32)
        canvas_width, canvas_height = self.size
        upper_bounds[:self._node_num] = canvas_width
        upper_bounds[self._node_num:2*self._node_num] = canvas_height
        upper_bounds[2*self._node_num:] = canvas_width * canvas_height # sqrt(x^2+y^2) <= x*y
        return upper_bounds
