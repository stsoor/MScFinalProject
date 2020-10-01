from problem.model import Hypergraph, NodewiseDistanceModel
from utilities.random import RandomGenerator
from solution.evaluation import DistanceModelEvaluator
from solution.drawing import HypergraphDrawer
from solution.initializer import DistanceSolutionInitializer
from optimization.algorithm import PSO
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

h = h1

x = np.array([100,100,200,200,120,150,180,100,200,200,100,120,150,120] + [50] * 7, dtype=np.float32)

problem = NodewiseDistanceModel(h, 10, 1080, 720)
components = problem.get_edge_components(x)

#print(components)

evaluator = DistanceModelEvaluator(problem)
#print(evaluator(x))
initializer = DistanceSolutionInitializer
pso = PSO(problem, initializer, evaluator, 100, 0.5, 0.5, 0.5, 100)
best_global_value, best_global_position, iteration = pso.run()
print(best_global_value, iteration)

drawer = HypergraphDrawer(problem, best_global_position)
img = drawer([[0,255,0],[255,0,0]])
cv2.imshow('Window', img)
cv2.waitKey(0)

# initializer = DistanceModelInitializer

# pso = PSO(evaluator, initializer, 5, 0.5, 0.5, 0.5, 100)
# best_global_value, best_global_position, iteration = pso.run()

# drawer = HypergraphDrawer(problem, components)
# img = drawer()
#img = drawer([[0,255,0],[255,0,0]])
# cv2.imshow('Window', img)
# cv2.waitKey(0)
