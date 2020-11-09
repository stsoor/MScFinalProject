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
np.random.seed(101)
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
#alg = HypergraphPSO(problem, initializer, evaluator, 100, 0.5, 0.5, 0.5, 100, debug=20)
#alg = NaiveMultiRowHypergraphGA(problem, initializer, evaluator, population_size, 0.2, 0.3, RandomGenerator('normal', 0, 3), 100, debug=20)
alg = EdgewiseHypergraphGA(problem, initializer, evaluator, population_size, 0.2, 0.3, RandomGenerator('normal', 0, 3), 100, debug=20)
best_global_value, best_global_position, iteration = alg()
print(best_global_value, iteration)

drawer = HypergraphDrawer(problem, best_global_position)
#drawer.show(colors=[[0,255,0],[255,0,0]])
drawer.show()

# initializer = DistanceModelInitializer

# pso = PSO(evaluator, initializer, 5, 0.5, 0.5, 0.5, 100)
# best_global_value, best_global_position, iteration = pso()

# drawer = HypergraphDrawer(problem, components)
# img = drawer()
#img = drawer([[0,255,0],[255,0,0]])
# cv2.imshow('Window', img)
# cv2.waitKey(0)
