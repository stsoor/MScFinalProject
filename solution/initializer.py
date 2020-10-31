import numpy as np
import networkx as nx

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

class SpringDistanceSolutionInitializer(DistanceSolutionInitializer):
    def __new__(self, problem, dimensionality):

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

        return x

class EdgewiseRandomDistanceSolutionInitializer(DistanceSolutionInitializer):
    def __new__(self, problem, dimensionality):
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

        return x
