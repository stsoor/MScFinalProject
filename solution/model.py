import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
import pandas as pd

class SolutionModel:
    def get_edge_components(self):
        pass
    def get_positions(self):
        pass

class NodewiseDistanceSolutionModel(SolutionModel):
    def __init__(self, problem_model):
        self._problem_model = problem_model
        self._node_num, self._edge_num = self._problem_model.hypergraph.shape
        self._x = np.zeros((self._node_num * 3,), dtype=np.float32) # [x_1,...,x_(node_num),y_1,...,y_(node_num),d_1,...,d_(node_num)]

    def _calculate_distance_matrix(self):
        positions = self.get_positions()
        return distance_matrix(positions, positions, threshold=10**8) # pdist would give a more compact distance descriptor

    def _calculate_min_thresholds(self):
        distance_thresholds = self._x[self._node_num*2:]
        return np.minimum.outer(distance_thresholds, distance_thresholds) # (for np.minimum.outer) pylint: disable=no-member

    def _is_in_threshold(self):
        return self._calculate_distance_matrix() <= self._calculate_min_thresholds()

    # def get_edge_components(self):
    #     all_components = []
    #     non_diagonal_mask = ~np.diag(np.full(self._node_num, True, dtype=np.bool))
    #     hypergraph = self._problem_model.hypergraph
    #     in_threshold_matrix = self._is_in_threshold()
    #     for edge_id in range(self._edge_num):
    #         edge = hypergraph[:, edge_id]
    #         edge_adjacency = np.outer(edge, edge) & non_diagonal_mask & in_threshold_matrix
    #         edge_graph = nx.from_numpy_matrix(edge_adjacency)
    #         edge_graph.remove_nodes_from(np.where(~edge)[0])
    #         edge_components = list(nx.connected_components(edge_graph))
    #         all_components.append(edge_components)
        
    #     return all_components

    def get_edge_components(self):
        all_components = []
        # non_diagonal_mask = ~np.diag(np.full(self._node_num, True, dtype=np.bool))
        hypergraph = self._problem_model.hypergraph
        in_threshold_matrix = self._is_in_threshold()
        for edge_id in range(self._edge_num):
            edge = hypergraph[:, edge_id]
            edge_nodes = np.where(edge)[0]
            # index_mesh = np.ix_(edge_nodes, edge_nodes)
            # edge_non_diagonal_mask = non_diagonal_mask[index_mesh]
            # edge_in_threshold_matrix = in_threshold_matrix[index_mesh]
            # edge_non_diagonal_mask = non_diagonal_mask[edge_nodes,:][:,edge_nodes]
            edge_non_diagonal_mask = ~np.diag(np.full(edge_nodes.size, True, dtype=np.bool))
            edge_in_threshold_matrix = in_threshold_matrix[edge_nodes,:][:,edge_nodes]
            edge_adjacency = edge_non_diagonal_mask & edge_in_threshold_matrix
            labels = edge_nodes
            edge_pd_adjacency = pd.DataFrame(edge_adjacency, index=labels, columns=labels)
            edge_graph = nx.from_pandas_adjacency(edge_pd_adjacency)
            edge_components = list(nx.connected_components(edge_graph))
            all_components.append(edge_components)
        
        return all_components  

    def get_positions(self):
        return self._x[:self._node_num*2].reshape((self._node_num, 2), order='F')

    def get_vector_view(self):
        return self._x

    def get_problem(self):
        return self._problem_model

    def get_vector_lower_bounds(self):
        return np.full(self._x.size, 0.0, dtype=np.float32)

    def get_vector_upper_bounds(self):
        upper_bounds = np.empty(self._x.size, dtype=np.float32)
        canvas_width, canvas_height = self._problem_model.size
        upper_bounds[:self._node_num] = canvas_width
        upper_bounds[self._node_num:2*self._node_num] = canvas_height
        upper_bounds[2*self._node_num:] = canvas_width * canvas_height # sqrt(x^2+y^2) <= x*y
        return upper_bounds
