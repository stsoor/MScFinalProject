import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix
import copy

class Hypergraph:
    def __new__(self, node_num, edge_num):
        return np.full((node_num, edge_num), False, dtype=np.bool)
    
    def __getitem__(self, key): # to hide linter errors
        pass
    def __setitem__(self, key, value): # to hide linter errors
        pass

class RandomHypergraph:
    def __new__(self, node_num, edge_num, random_generator, inclusion_threshold):
        hypergraph = random_generator((node_num, edge_num)) <= inclusion_threshold
        contained_node_num = np.sum(hypergraph, axis=0)
        edges_to_override = np.where(contained_node_num == 0)
        selected_nodes = np.random.randint(0, node_num, edge_num)
        hypergraph[selected_nodes[edges_to_override], edges_to_override] = True
        return hypergraph


class ProblemModel:
    def __init__(self, hypergraphs, min_node_distance, canvas_width, canvas_height):
        self._hypergraphs = hypergraphs if isinstance(hypergraphs, list) else [hypergraphs]
        assert(min_node_distance > 0)
        self.min_node_distance = min_node_distance
        self.size = (canvas_width, canvas_height)
    
    @property
    def hypergraphs(self):
        return self._hypergraphs

    @hypergraphs.setter
    def hypergraphs(self, new_hypergraphs):
        self._hypergraphs = new_hypergraphs

    @property
    def hypergraph(self):
        if len(self.hypergraphs) == 1:
            return self.hypergraphs[0]
        raise ValueError
    
    @hypergraph.setter
    def hypergraph(self, new_hypergraph):
        if (isinstance(new_hypergraph, list) and len(new_hypergraph) == 0):
            self.hypergraphs = new_hypergraph
        elif (isinstance(new_hypergraph, (np.ndarray, np.generic)) and len(new_hypergraph.shape) == 2):
            self.hypergraphs = [new_hypergraph]
        else:
            raise ValueError

    def clone(self, copy_hypergraphs=False):
        return ProblemModel(*self.dump_as_args(copy_hypergraphs))
    
    def dump_as_args(self, copy_hypergraphs=False):
        return ((copy.deepcopy(self.hypergraphs) if copy_hypergraphs else []), self.min_node_distance, self.size[0], self.size[1])

class NodewiseDistanceModel(ProblemModel):
    def __init__(self, hypergraphs, min_node_distance, canvas_width, canvas_height):
        super(NodewiseDistanceModel, self).__init__(hypergraphs, min_node_distance, canvas_width, canvas_height)

    def clone(self, copy_hypergraphs=False):
        return NodewiseDistanceModel(*self.dump_as_args(copy_hypergraphs))
    
    def clone_subproblem(self, problem, kept_hypergraph_index_list):
        subproblem = problem.clone(False)
        subproblem.hypergraph = [problem.hypergraphs[idx].copy() for idx in kept_hypergraph_index_list]
        return subproblem

    def calculate_distance_matrix(self, x):
        positions = self.extract_positions2D(x)
        return distance_matrix(positions, positions, threshold=10**8) # pdist would give a more compact distance descriptor

    def _get_distance_thresholds(self, x):
        node_num = self.hypergraphs[0].shape[0]
        distance_thresholds = x[node_num*2:]
        return distance_thresholds

    def _calculate_min_thresholds(self, x):
        distance_thresholds = self._get_distance_thresholds(x)
        return np.minimum.outer(distance_thresholds, distance_thresholds) # (for np.minimum.outer) pylint: disable=no-member

    def _is_in_threshold(self, x):
        return self.calculate_distance_matrix(x) <= self._calculate_min_thresholds(x)

    def extract_positions2D(self, x):
        node_num = self.hypergraphs[0].shape[0]
        assert(len(x) == node_num*3)
        return x[:node_num*2].reshape((node_num, 2), order='F')

    def get_edge_components(self, x):
        node_num, edge_num = self.hypergraphs[0].shape
        assert(len(x) == node_num*3)
        all_components = []
        in_threshold_matrix = self._is_in_threshold(x)
        for edge_id in range(edge_num):
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
    
    def get_hypergraph_as_graph(self, weighted=False):
        if not weighted:
            incidence = (self.hypergraph @ self.hypergraph.T)
            np.fill_diagonal(incidence, False)
        else:
            hypergraph_as_int = self.hypergraph.astype(np.int32)
            incidence = (hypergraph_as_int @ hypergraph_as_int.T)
            np.fill_diagonal(incidence, 0)
        return nx.from_numpy_matrix(incidence)

    def get_vector_lower_bounds(self):
        node_num = self.hypergraphs[0].shape[0]
        return np.full(node_num*3, 0.0, dtype=np.float32)

    def get_vector_upper_bounds(self):
        node_num = self.hypergraphs[0].shape[0]
        upper_bounds = np.empty(node_num*3, dtype=np.float32)
        canvas_width, canvas_height = self.size
        upper_bounds[:node_num] = canvas_width
        upper_bounds[node_num:2*node_num] = canvas_height
        upper_bounds[2*node_num:] = canvas_width * canvas_height # sqrt(x^2+y^2) <= x*y
        return upper_bounds
