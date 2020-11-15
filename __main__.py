from problem.model import Hypergraph, RandomHypergraph, NodewiseDistanceModel
from utilities.random import RandomGenerator
from solution.evaluation import DistanceModelEvaluator
from solution.drawing import HypergraphDrawer
from solution.initializer import DistanceSolutionInitializer, EdgewiseRandomDistanceSolutionInitializer, SpringDistanceSolutionInitializer, InitializerBlender
from optimization.algorithm.particle_swarm_optimization import HypergraphPSO
from optimization.algorithm.genetic_algorithm import NaiveMultiRowHypergraphGA, EdgewiseHypergraphGA
from utilities.callable_coupling import CallableCoupling
import cv2
import numpy as np

h1 = Hypergraph(7, 2)
h2 = Hypergraph(7, 2)
h3 = Hypergraph(6, 2)
h4 = Hypergraph(10, 4)
h5 = Hypergraph(10, 2)
h6 = Hypergraph(3, 2)
h7 = Hypergraph(8, 2)
h8 = Hypergraph(8, 2)
h1[:4,0] = True
h1[4:,1] = True
#h1[:,0] = True
h2[:4,0] = True
h2[4:,1] = True
h3[:3,0] = True
h3[3:,1] = True
h4[:4,0] = True
h4[4,1] = True
h4[5:7,2] = True
h4[7:,3] = True
h5[:6,0] = True
h5[6:,1] = True
h6[0,0] = True
h6[1:,1] = True
h7[:4,0] = True
h7[4:,1] = True
h8[:4,0] = True
h8[4:,1] = True
#np.random.seed(101)
h9 = RandomHypergraph(20, 8, RandomGenerator('uniform', 0, 1), 0.2)

h = h9

#x = np.array([100,100,200,200,120,150,180,100,200,200,100,120,150,120] + [50] * 7, dtype=np.float32)

problem = NodewiseDistanceModel(h, 10, 1080, 720)
#components = problem.get_edge_components(x)

#print(components)

evaluator = DistanceModelEvaluator(intersection_measure_weight = 1.0, debug=None)
evaluator = DistanceModelEvaluator(
                 edge_count_weight = 1000.0,
                 circularity_weight = 10.0,
                 not_missing_containment_weight = 10000.0,
                 not_miscontained_weight = 10000.0,
                 no_single_separation_weight = 1000.0,
                 min_distance_weight = 100.0,
                 nodes_at_min_distance_weight = 10.0,
                 area_proportionality_weight = 10.0,
                 intersection_measure_weight = 1000.0,
                 debug=None)
evaluator = CallableCoupling(evaluator, problem, True, _add_call_args_before=True)

#print(evaluator(x))
population_size = 100
#initializer = DistanceSolutionInitializer
initializer = EdgewiseRandomDistanceSolutionInitializer
#initializer = SpringDistanceSolutionInitializer
initializer = CallableCoupling(initializer, problem, population_size, use_closest_node_distance=False)

#initializer = DistanceSolutionInitializer
#initializer = SpringDistanceSolutionInitializer
#initializer = InitializerBlender([SpringDistanceSolutionInitializer, EdgewiseRandomDistanceSolutionInitializer], [0.7,0.3])
#alg = HypergraphPSO(problem.get_vector_lower_bounds(), problem.get_vector_upper_bounds(), initializer, evaluator, 100, 0.5, 0.5, 0.5, 100, debug=20, problem=problem)
#alg = NaiveMultiRowHypergraphGA(initializer, evaluator, population_size, 0.2, 0.3, RandomGenerator('normal', 0, 3), 100, debug=20, problem=problem)

#alg = EdgewiseHypergraphGA(initializer, evaluator, population_size, 0.2, 0.3, RandomGenerator('normal', 0, 3), 100, debug=20, problem=problem)
#best_global_value, best_global_position, iteration = alg()
#print(best_global_value, iteration)
#
#drawer = HypergraphDrawer(problem, best_global_position)
#drawer.show()

def get_laplacian_graph(hypergraph):
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

def get_laplacian(hypergraph):
    node_num, _edge_num = hypergraph.shape
    G_x = get_laplacian_graph(hypergraph)
    vertex_degrees = (G_x > 0).sum(axis=1)
    nonzero_division_vertex_degrees = vertex_degrees.copy().reshape((-1,1))
    nonzero_division_vertex_degrees[nonzero_division_vertex_degrees == 0] = 1
    A_x = G_x / nonzero_division_vertex_degrees
    D_inv_root = 1 / nonzero_division_vertex_degrees
    D_inv_root[vertex_degrees.reshape((-1,1)) == 0] = 0
    D_inv_root = np.diag(D_inv_root.flatten())
    D_inv_root = np.sqrt(D_inv_root)

    return np.eye(node_num) - D_inv_root @ A_x @ D_inv_root

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def gcn_layer(G, X, W, activation=relu):
    return activation(G @ X @ W)

laplacian = get_laplacian(problem.hypergraph)
g = get_laplacian_graph(problem.hypergraph)


input_num = problem.hypergraph.shape[0]
node_num = 20

I = np.eye(node_num)
W_1 = np.random.normal(loc=0, scale=1, size=(node_num, 4))
W_2 = np.random.normal(loc=0, scale=1, size=(W_1.shape[1], 2))

H_1 = gcn_layer(laplacian, I, W_1, relu)
H_2 = gcn_layer(laplacian, H_1, W_2, sigmoid)
output = H_2

output[:,0] *= 1080
output[:,1] *= 720
positions = np.hstack((output.flatten(order='F'),np.full(20, np.inf)))

drawer = HypergraphDrawer(problem, positions)
print(evaluator(positions)[0])
drawer.show()

alma = 8