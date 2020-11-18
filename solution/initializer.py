import numpy as np
import networkx as nx
from enum import Enum

class Initializer:
    def __new__(self, problem, dimensionality):
        pass

class DistanceStatistic(Enum):
    NONE = 0
    MIN = 1
    MAX = 2
    MEAN = 3

class DistanceSolutionInitializer(Initializer):

    def __new__(self, problem, dimensionality, used_node_distance=DistanceStatistic.NONE):

        node_num = problem.hypergraph.shape[0]
        lower_bounds = problem.get_vector_lower_bounds()
        upper_bounds = problem.get_vector_upper_bounds()
        bounds_range = upper_bounds - lower_bounds

        x = np.random.uniform(0, 1, size=(dimensionality, node_num*3)) # current particle positions
        x = lower_bounds + x * bounds_range
        x = self._apply_node_distance_override(x, problem, used_node_distance)

        return x

    @staticmethod
    def _apply_node_distance_override(x, problem, used_node_distance):
        def override_row_with_closest_node_distance(x_row, problem, used_node_distance):
            distance_matrix = problem.calculate_distance_matrix(x_row)
            max_distance = np.max(distance_matrix)
            is_neighbour_table = (problem.hypergraph @ problem.hypergraph.T)
            distance_matrix = distance_matrix * is_neighbour_table
            distance_matrix[~is_neighbour_table] = max_distance
            np.fill_diagonal(distance_matrix, max_distance)
            if used_node_distance == DistanceStatistic.MIN:
                closest_neighbour_distance = distance_matrix.min(axis=1).flatten()
            elif used_node_distance == DistanceStatistic.MAX:
                closest_neighbour_distance = distance_matrix.max(axis=1).flatten()
            elif used_node_distance == DistanceStatistic.MEAN:
                closest_neighbour_distance = distance_matrix.mean(axis=1).flatten()
            else:
                raise ValueError
            node_num = problem.hypergraph.shape[0]
            
            new_row = np.hstack((x_row[:2*node_num], closest_neighbour_distance))
            return new_row

        if used_node_distance != DistanceStatistic.NONE:
            x = np.apply_along_axis(override_row_with_closest_node_distance, 1, x, problem, used_node_distance)
        return x


class SpringDistanceSolutionInitializer(DistanceSolutionInitializer):
    def __new__(self, problem, dimensionality, used_node_distance=DistanceStatistic.NONE):

        def build_x_row(G, node_num):
            position_dict = nx.spring_layout(nx_graph)
            x_list = []
            for node_id in range(node_num):
                x_list.append(position_dict[node_id])
            return np.stack(x_list).reshape((-1,),order='F')

        node_num = problem.hypergraph.shape[0]
        nx_graph = problem.get_hypergraph_as_graph()
        positions = np.apply_along_axis(lambda row_id, G, node_num: build_x_row(G, node_num), 1, np.arange(dimensionality).reshape(dimensionality,1), nx_graph, node_num)
#        positions = positions.reshape(dimensionality, 2*node_num) # removing excess (1-point) dimension

        lower_bounds = problem.get_vector_lower_bounds()
        upper_bounds = problem.get_vector_upper_bounds()
        bounds_range = upper_bounds - lower_bounds

        positions = (positions + 1) / 2
        positions *= bounds_range[:2*node_num]

        d_values = np.random.uniform(0, 1, size=(dimensionality, node_num))
        d_values = lower_bounds[2*node_num:] + d_values * bounds_range[2*node_num:]

        x = np.hstack((positions, d_values))
        x = self._apply_node_distance_override(x, problem, used_node_distance)

        return x

class EdgewiseRandomDistanceSolutionInitializer(DistanceSolutionInitializer):
    def __new__(self, problem, dimensionality, used_node_distance=DistanceStatistic.NONE):
        epsilon = 1e-8

        node_num, edge_num = problem.hypergraph.shape
        lower_bounds = problem.get_vector_lower_bounds()
        upper_bounds = problem.get_vector_upper_bounds()
        assert np.all(np.mean(lower_bounds[:node_num]) == lower_bounds[:node_num]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        assert np.all(np.mean(lower_bounds[node_num:2*node_num]) == lower_bounds[node_num:2*node_num]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        assert np.all(np.mean(lower_bounds[2*node_num:]) == lower_bounds[2*node_num:]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        assert np.all(np.mean(upper_bounds[:node_num]) == upper_bounds[:node_num]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        assert np.all(np.mean(upper_bounds[node_num:2*node_num]) == upper_bounds[node_num:2*node_num]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        assert np.all(np.mean(upper_bounds[2*node_num:]) == upper_bounds[2*node_num:]), 'edgewise initializer can only be used if all nodes have the same lower/upper bounds'
        bounds_range = upper_bounds - lower_bounds

        edge_x_values = np.random.uniform(0, 1, size=(dimensionality, edge_num))
        edge_y_values = np.random.uniform(0, 1, size=(dimensionality, edge_num))
        node_d_values = np.random.uniform(0, 1, size=(dimensionality, node_num))

        edge_x_values = lower_bounds[0] + edge_x_values * bounds_range[0]
        edge_y_values = lower_bounds[node_num] + edge_y_values * bounds_range[node_num]
        node_d_values = lower_bounds[2*node_num] + node_d_values * bounds_range[2*node_num]

        node_containment_counts = np.sum(problem.hypergraph, axis=1)
        node_containment_counts[node_containment_counts == 0] = 1

        get_average_position = lambda values_row, hypergraph, node_containment_counts: np.sum(values_row * hypergraph, axis=1) / node_containment_counts
        node_x_values = np.apply_along_axis(get_average_position, 1, edge_x_values, problem.hypergraph, node_containment_counts)
        node_y_values = np.apply_along_axis(get_average_position, 1, edge_y_values, problem.hypergraph, node_containment_counts)

        node_x_values_mask = node_x_values < epsilon
        node_x_values[node_x_values_mask] = np.random.uniform(lower_bounds[0], upper_bounds[0], size=node_x_values_mask.sum())
        node_y_values_mask = node_y_values < epsilon
        node_y_values[node_y_values_mask] = np.random.uniform(lower_bounds[node_num], upper_bounds[node_num], size=node_y_values_mask.sum())

        x = np.hstack((node_x_values, node_y_values, node_d_values))
        x = self._apply_node_distance_override(x, problem, used_node_distance)

        return x

class InitializerBlender:
    def __init__(self, initializers, weights):
        self.initializers = initializers
        self.weights = np.array(weights)

        assert len(initializers) == len(weights), 'not the same number of elements in initializers and weights'
        assert np.abs(self.weights.sum() - 1) <= 1e-8, 'sum of weights is not 1'
    
    def __call__(self, problem, dimensionality):
        selected_x_parts = []
        selected_row_num_sum = 0
        for i in range(len(self.initializers)):
            x = self.initializers[i](problem, dimensionality)
            row_num = x.shape[0]
            selected_row_num = int(row_num * self.weights[i]) if i != len(self.initializers) - 1 else row_num - selected_row_num_sum
            row_indices = np.arange(row_num)
            np.random.shuffle(row_indices)
            selected_row_ids = row_indices[:selected_row_num]
            selected_x_parts.append(x[selected_row_ids,:])
            selected_row_num_sum += selected_row_num
        x = np.vstack(selected_x_parts)
        return x
