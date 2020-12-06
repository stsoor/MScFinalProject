import unittest

from problem.model import Hypergraph, RandomHypergraph, NodewiseDistanceModel
from utilities.random import RandomGenerator
from solution.evaluation import DistanceModelEvaluator
from solution.drawing import HypergraphDrawer
from solution.initializer import DistanceSolutionInitializer
import cv2
import numpy as np

# class DistanceModelEvaluatorTest(unittest.TestCase):
#     def setUp(self):
#         self._init_h_1()
    
#     def _init_h_1(self):
#         h_1 = Hypergraph(7, 2)
#         h_1[:4,0] = True
#         h_1[4:,1] = True

#         self.h_1 = h_1
#         self.problem_1 = NodewiseDistanceModel(h_1, 10, 1080, 720)

#     def _build_edge_count_only_evaluator(self, problem):
#         evaluator = DistanceModelEvaluator(
#                         edge_count_weight = 1,
#                         circularity_weight = 0,
#                         not_missing_containment_weight = 0,
#                         not_miscontained_weight = 0,
#                         edge_segment_size_weight = 0,
#                         min_distance_weight = 0,
#                         min_distance_occurence_weight = 0,
#                         area_proportionality_weight = 0,
                #  all_intersection_measure_weight = 1000.0,
                #  invalid_intersection_weight= 1000.0,
#                         debug=None)
#         return evaluator
    
#     def test_edge_count(self):
#         evaluator = self._build_edge_count_only_evaluator(self.problem_1)
#         x = np.array([100,100,200,200,120,150,180] + [100,200,200,100,120,150,120] + [50] * 7, dtype=np.float32)
#         value = evaluator(x, self.problem_1)
#         self.assertAlmostEqual(value, 0.375)


if __name__ == '__main__':
    unittest.main()
