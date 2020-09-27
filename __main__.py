from problem.model import ProblemModel, Hypergraph
from utilities.random import RandomGenerator
from solution.model import NodewiseDistanceSolutionModel


h = Hypergraph(8, 2)
h[[0,3,4],0] = True
h[[1,2,3,4,5,6],1] = True
problem = ProblemModel(h, 10, 1080, 720)
solution_model = NodewiseDistanceSolutionModel(problem)
x = solution_model.get_vector_view()
#x[16:] = 1.0
x[0] = -5
x[1] = -4
x[2] = -3
x[3] = -5
x[4] = -4
x[5] = -3
x[6] = -5
x[7] = 1
x[8] = 6.5
x[9] = 6.2
x[10] = 7
x[11] = 6
x[12] = 6
x[13] = 6
x[14] = 5.8
x[15] = 1
x[16] = 1.25
x[17] = 0.3
x[18] = 1.1
x[19] = 0.5
x[20] = 1.25
x[21] = 1
x[22] = 0.5
x[23] = 200
components = solution_model.get_edge_components()

print(8)
