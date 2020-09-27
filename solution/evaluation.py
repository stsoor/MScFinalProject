import numpy as np
from scipy.spatial.distance import pdist
import cv2

class Evaluator:
    class Score:
        def __init__(self, edgwise_scores, final_score):
            self.edgwise_scores = edgwise_scores
            self.final_score = final_score
    
        def get_score(self, edge_id=None):
            return self.final_score if edge_id is None else self.edgwise_scores[edge_id]

    def get_score(self, solution):
        pass

class DistanceModelEvaluator(Evaluator):
    def __init__(self,
                 edge_count_weight,
                 circularity_weight,
                 not_missing_containment_weight,
                 not_miscontained_weight,
                 no_single_separation_weight,
                #  spatial_dispersion_weight,
                 min_distance_weight,
                 nodes_at_min_distance_weight,
                 area_proportionality_weight):
        self.edge_count_weight              = edge_count_weight
        self.circularity_weight             = circularity_weight
        self.not_missing_containment_weight = not_missing_containment_weight
        self.not_miscontained_weight        = not_miscontained_weight
        self.no_single_separation_weight    = no_single_separation_weight
        # self.spatial_dispersion_weight      = spatial_dispersion_weight
        self.min_distance_weight            = min_distance_weight
        self.nodes_at_min_distance_weight   = nodes_at_min_distance_weight
        self.area_proportionality_weight    = area_proportionality_weight

    # best [0,1] worst
    def _calculate_nodes_not_missing_ratio(self): # O(m)
        # node_num = solution.get_problem().hypergraph.shape[0]
        # return node_num / node_num # each vertex will be contained using the convex hull algorithm
        return 0.0

    def _get_convex_hull(self, edge_segment, all_positions): # O(S*d*log(d)) <= O(n^2*log(n))
        mask = list(edge_segment)
        hull = cv2.convexHull(all_positions[mask].astype(np.float32))
        hull = hull.reshape((hull.shape[0], 2))
        return hull
    
    def _get_area(self, convex_hull, r):
        if convex_hull.shape[0] > 2:
            return cv2.contourArea(convex_hull)
        elif convex_hull.shape[0] == 2:
            start, end = convex_hull
            rectangle_length = np.linalg.norm(start-end)
            return ((r ** 2 * np.pi) + (rectangle_length * 2.0 * r))
        else:
            return r*r*np.pi


    # best [0,1] worst
    def _calculate_circularity_measure(self, convex_hull, r): # O(S) <= O(n)
        # 4*pi*Area / Perimeter^2 : [0,1] range, 1 is circle, ~0 has a lot of angles
        # TODO weight it by contained nodes?
        if convex_hull.shape[0] == 1:
            return 0.0

        area = self._get_area(convex_hull, r)
        if convex_hull.shape[0] > 2:
            return 1.0 - 4.0 * np.pi * area / cv2.arcLength(convex_hull, True)**2
        elif convex_hull.shape[0] == 2:
            start, end = convex_hull
            rectangle_length = np.linalg.norm(start-end)
            return 1.0 - (4 * np.pi * area) / (2.0 * r * np.pi  + 2.0 * rectangle_length)**2
    
    # best [0, 1) worst
    def _calculate_miscontained_nodes_ratio(self, segment_vertex_ids, convex_hull, all_positions, r): # O(S*((n-d)*d + d*log(d))) <= O(n^3)
        def calculate_two_point_rectangle(start, end, r):
            #v = np.array([end[0]-start[0], end[1]-start[1]], dtype=np.float32)
            n = np.array([start[1]-end[1], end[0]-start[0]], dtype=np.float32)
            n /= np.linalg.norm(n)

            c1 = (start + r*n)
            c2 = (start - r*n)
            c3 = (end - r*n)
            c4 = (end + r*n)
            return np.vstack([c1, c2, c3, c4]).astype(np.float32)
    
        miscontained_vertex_num = 0
        #segment_vertex_num = 0
        for vertex_id in range(all_positions.shape[0]):
            if vertex_id in segment_vertex_ids:
                #segment_vertex_num += 1
                continue

            vertex = all_positions[vertex_id]
            if convex_hull.shape[0] > 2:
                if cv2.pointPolygonTest(convex_hull.astype(np.float32), tuple(vertex), False) >= 0:
                    miscontained_vertex_num += 1
            elif convex_hull.shape[0] == 2:
                start, end = convex_hull
                rectangle_corners = calculate_two_point_rectangle(start, end, r)
                if np.linalg.norm(start - vertex) <= r or np.linalg.norm(end - vertex) <= r or cv2.pointPolygonTest(rectangle_corners.astype(np.float32), tuple(vertex), False) >= 0:
                    miscontained_vertex_num += 1
            else:
                if np.linalg.norm(convex_hull[0] - vertex) < r:
                    miscontained_vertex_num += 1
        
        return miscontained_vertex_num / all_positions.shape[0]

    # best [0,1) worst
    def _calculate_edge_segments_num_ratio(self, edge_segments):
        #return len(edge_components) / sum(map(len, edge_components))
        return 1.0 - 1.0 / len(edge_segments)

    # best [0, 1] worst
    def _calculate_single_node_separations_ratio(self, edge_segments): # O(S) <= O(n)
        one_long_segment_num = sum([1 for segment in edge_segments if len(edge_segments) == 1])
        if len(edge_segments) > 1:
            return one_long_segment_num / len(edge_segments)
        else:
            return 0.0

    # def _measure_spatial_dispersion(self, solution): # O(n)
    #     # test if it is from a uniform distribution
    #     # check if the sum of x & y differences to the center are 0

    # best [0,1] worst
    def _calculate_area_proportionality(self, edge_id, edge_hulls, solution, r): # O(S) <= O(n)
        edge_area = sum([self._get_area(convex_hull, r) for convex_hull in edge_hulls])
        canvas_size = solution.get_problem().size[0] * solution.get_problem().size[1]
        all_vertex_num = solution.get_problem().hypergraph.shape[0]
        edge_vertex_num = solution.get_problem().hypergraph[:, edge_id].sum()
        return abs(edge_area / canvas_size - edge_vertex_num / all_vertex_num) # abs([0,1] - [0,1])

    # [0,sqrt(width^2+height^2)], [0, n]
    def _calculate_min_node_distance_and_occurences(self, solution, positions): # O(n^2), this measure is global
        distances = pdist(positions)
        min_distance = distances.min()
        num_mins = (distances == min_distance).sum()
        return min_distance, num_mins

    # best [0,1] worst
    def _calculate_min_node_distance_to_target_ratio(self, solution, min_distance_realisation): # O(n^2), this measure is global
        return 1.0 - max(1.0, min_distance_realisation) / solution.get_problem().min_node_distance

    def _get_r(self, solution):
        r = solution.get_problem().min_node_distance / 2.0
        return r

    def get_score(self, solution):
        all_edge_components = solution.get_edge_components()
        edgewise_scores = np.full((1,len(all_edge_components)), 1.0)
        all_positions = solution.get_positions()

        r = self._get_r(solution)

        for edge_id in range(len(all_edge_components)):
            segment_hulls = [self._get_convex_hull(segment, all_positions) for segment in all_edge_components[edge_id]]
            edge = solution.get_problem().hypergraph[:, edge_id]
            segment_vertex_ids = np.where(edge)[0]

            nodes_not_missing_measure = self._calculate_nodes_not_missing_ratio()
            circularity_measure = sum([self._calculate_circularity_measure(segment_hull, r) / len(segment_hulls) for segment_hull in segment_hulls])
            miscounted_nodes_measure = sum([self._calculate_miscontained_nodes_ratio(segment_vertex_ids, segment_hull, all_positions, r) / len(segment_hulls) for segment_hull in segment_hulls])
            edge_segment_num_measure = self._calculate_edge_segments_num_ratio(all_edge_components[edge_id])
            area_proportionality_measure = self._calculate_area_proportionality(edge_id, segment_hulls, solution, r)

            a = 4 # TODO delete
            # TODO edgewise_scores = 

        min_node_distance, min_distance_occurence_measure = self._calculate_min_node_distance_and_occurences(solution, all_positions)
        min_distance_measure = self._calculate_min_node_distance_to_target_ratio(solution, min_node_distance)

        a = 4 # TODO delete
        #TODO return

        #return self.Score(edgewise_scores, ???)