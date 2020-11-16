import numpy as np

from enum import Enum
import copy

from optimization.algorithm.genetic_algorithm import NaiveGA
from solution.evaluation import HGCNEvaluator
from utilities.callable_coupling import CallableCoupling

class HypergraphConvolutionalNetwork:
    class Activation(Enum):
        Sigmoid = 1
        Relu = 2
        Softmax = 3

    def __init__(self, neuron_num_sequence, activation_function_sequence): #input is represented as the first layer
        assert len(neuron_num_sequence) >= 2
        assert len(neuron_num_sequence) == len(activation_function_sequence) + 1
        self.neuron_num_sequence = neuron_num_sequence
        self.activation_function_sequence = activation_function_sequence
        self.weights = []
        self.activations = []
        self.is_trained = False
        #self._initialize_parameters()
    
    def clone(self, clone_weights_and_activations=True):
        other = HypergraphConvolutionalNetwork(self.neuron_num_sequence, self.activation_function_sequence)
        if clone_weights_and_activations:
            other.weights = copy.deepcopy(self.weights)
            other.activations = copy.deepcopy(self.activations)
            other.is_trained = self.is_trained
        return other

    def _sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def _initialize_parameters(self, initializer=lambda shape: np.random.normal(0,1,shape)):
        self.weights = [initializer((self.neuron_num_sequence[i], self.neuron_num_sequence[i+1])) for i in range(len(self.neuron_num_sequence) - 1)]
        self.activations = [self._resolve_activation_enum(activation_enum_value) for activation_enum_value in self.activation_function_sequence]
        self.is_trained = False
    
    def reset(self, value=None):
        self._initialize_parameters(initializer=lambda shape: np.full(shape, value))

    def _resolve_activation_enum(self, enum_value):
        if enum_value == self.Activation.Sigmoid:
            return self._sigmoid
        elif enum_value == self.Activation.Relu:
            return self._relu
        elif enum_value == self.Activation.Softmax:
            return self._softmax
        else:
            raise NotImplementedError('not implemented activation function is requested')

    def dump_row_vector(self):
        def copy_to_vector(vector, np_array_sequence, start_index=0):
            for item in np_array_sequence:
                length = item.size
                next_start_index = start_index + length
                vector[start_index:next_start_index] = item.flatten()
                start_index = next_start_index
            return start_index

        vector_size = sum(map(lambda x: x.size, self.weights))
        vector = np.empty(vector_size, dtype=np.float32)
        copy_to_vector(vector, self.weights, 0)
        return vector
    
    def load_row_vector(self, vector):
        def fill_from_vector(np_array_sequence, vector, vector_start_index=0):
            for i in range(len(np_array_sequence)):
                length = np_array_sequence[i].size
                next_start_index = vector_start_index + length
                values = vector[vector_start_index:next_start_index]
                np_array_sequence[i] = values.reshape(np_array_sequence[i].shape).copy()
                vector_start_index = next_start_index
            return vector_start_index
        
        self.reset(0)
        fill_from_vector(self.weights, vector, 0)
    
    def train(self, problem, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct=0.5, debug=None):
        laplacians = [self.get_laplacian(hypergraph) for hypergraph in problem.hypergraphs]
        hgcn_evaluator = HGCNEvaluator(*evaluator.dump_as_args())
        hgcn_evaluator = CallableCoupling(hgcn_evaluator, problem, self, laplacians, _add_call_args_before=True)
        self.reset(0)
        example_row = self.dump_row_vector()
        initializer = lambda dimensionality: np.random.normal(0,1,(dimensionality, example_row.size))
        lower_bounds = np.full_like(example_row, -1000) # won't happen with std normal distribution
        upper_bounds = np.full_like(example_row, 1000)
        optimizer = NaiveGA(lower_bounds, upper_bounds, initializer, hgcn_evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct, debug)
        best_global_value, best_global_position, iteration = optimizer()
        self.load_row_vector(best_global_position)
        self.is_trained = True
        return best_global_value, iteration

    def get_laplacian_graph(self, hypergraph):
        node_num, edge_num = hypergraph.shape
        edge_weights = 1 / hypergraph.sum(axis=0) # edge size cannot be 0 for the problem to be well formed
        X = np.random.uniform(0, 1, node_num)
        X_table = np.abs(np.subtract.outer(X, X))
        X_table += np.random.uniform(0, 0.00001, node_num) # deciding ties randomly
        np.fill_diagonal(X_table, -np.inf) # cannot select self loops
        G_edges = np.empty((edge_num, 2), dtype=np.int32)
        for edge_idx in range(edge_num):
            edge_mask = hypergraph[:, edge_idx]
            edge_X_table = X_table.copy()
            not_in_edge_mask = ~edge_mask
            edge_X_table[not_in_edge_mask, :] = -np.inf
            edge_X_table[:, not_in_edge_mask] = -np.inf
            argmax = np.argmax(edge_X_table)
            row = argmax // node_num
            col = argmax % node_num
            G_edges[edge_idx] = [row, col]
        G_x = np.zeros((node_num, node_num), dtype=np.float32)
        G_x[G_edges[:,0], G_edges[:,1]] = edge_weights
        vertex_degrees = (G_x > 0).sum(axis=1)
        WD = np.diag(vertex_degrees - G_x.sum(axis=1))
        G_x += WD

        return G_x

    def get_laplacian(self, hypergraph):
        node_num, _edge_num = hypergraph.shape
        G_x = self.get_laplacian_graph(hypergraph)
        vertex_degrees = (G_x > 0).sum(axis=1)
        nonzero_division_vertex_degrees = vertex_degrees.copy().reshape((-1,1))
        nonzero_division_vertex_degrees[nonzero_division_vertex_degrees == 0] = 1
        A_x = G_x / nonzero_division_vertex_degrees
        D_inv_root = 1 / nonzero_division_vertex_degrees
        D_inv_root[vertex_degrees.reshape((-1,1)) == 0] = 0
        D_inv_root = np.diag(D_inv_root.flatten())
        D_inv_root = np.sqrt(D_inv_root)

        return np.eye(node_num) - D_inv_root @ A_x @ D_inv_root
    
    def predict(self, problem, laplacian=None): # assumes that the final values are between 0 and 1
        laplacian = self.get_laplacian(problem.hypergraph) if laplacian is None else laplacian
        assert self.weights
        assert laplacian.shape[0] == laplacian.shape[1]
        assert laplacian.shape[1] == self.weights[0].shape[0]
        lower_bounds = problem.get_vector_lower_bounds()
        upper_bounds = problem.get_vector_upper_bounds()
        A_curr = np.eye(laplacian.shape[0])
        for i in range(len(self.weights)):
            A_prev = A_curr
            W_curr = self.weights[i]
            Z_curr = laplacian @ A_prev @ W_curr
            A_curr = np.apply_along_axis(self.activations[i], 1, Z_curr)
        A_shape = A_curr.shape
        A_curr = A_curr.flatten(order='F')
        if self.activation_function_sequence[-1] == self.Activation.Softmax:
            pass
        elif self.activation_function_sequence[-1] == self.Activation.Sigmoid:
            A_curr += 1
            A_curr /= 2
        elif self.activation_function_sequence[-1] == self.Activation.Relu:
            A_curr /= A_curr.max()
        else:
            raise ValueError
        A_curr *= (upper_bounds - lower_bounds)
        A_curr += lower_bounds
        A_curr = A_curr.reshape(A_shape, order='F')
        return A_curr
