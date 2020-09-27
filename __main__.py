from problem.model import ProblemModel, Hypergraph
from utilities.random import RandomGenerator
from solution.model import NodewiseDistanceSolutionModel
from solution.evaluation import DistanceModelEvaluator
from solution.drawing import HypergraphDrawer
import cv2

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
# h1[:,0] = True
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
h6[1,0] = True
h6[1:,1] = True
h7[:4,0] = True
h7[4:,1] = True
h8[:4,0] = True
h8[4:,1] = True

h = h1

problem = ProblemModel(h, 10, 1080, 720)
solution_model = NodewiseDistanceSolutionModel(problem)
x = solution_model.get_vector_view()

#h1
x[14:] = 2000
x[0] = 100
x[1] = 200
x[2] = 100
x[3] = 200
x[4] = 120
x[5] = 400 # 180
x[6] = 150
x[7] = 100
x[8] = 100
x[9] = 200
x[10] = 200
x[11] = 120
x[12] = 120
x[13] = 150

components = solution_model.get_edge_components()
evaluator = DistanceModelEvaluator(*[1.0]*8)
evaluator.get_score(solution_model)
drawer = HypergraphDrawer(solution_model, components)
img = drawer()
#img = drawer([[0,255,0],[255,0,0]])
cv2.imshow('Window', img)
cv2.waitKey(0)


# h = Hypergraph(8, 2)
# h[[0,3,4],0] = True
# h[[1,2,3,4,5,6],1] = True
# problem = ProblemModel(h, 10, 1080, 720)
# solution_model = NodewiseDistanceSolutionModel(problem)
# x = solution_model.get_vector_view()
# #x[16:] = 1.0
# x[0] = -5
# x[1] = -4
# x[2] = -3
# x[3] = -5
# x[4] = -4
# x[5] = -3
# x[6] = -5
# x[7] = 1
# x[8] = 6.5
# x[9] = 6.2
# x[10] = 7
# x[11] = 6
# x[12] = 6
# x[13] = 6
# x[14] = 5.8
# x[15] = 1
# x[16] = 1.25
# x[17] = 0.3
# x[18] = 1.1
# x[19] = 0.5
# x[20] = 1.25
# x[21] = 1
# x[22] = 0.5
# x[23] = 200
# components = solution_model.get_edge_components()

print(8)
