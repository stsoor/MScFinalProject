import numpy as np
from scipy.spatial.distance import pdist
import cv2

class Evaluator:
    def __call__(self, x):
        pass

class DistanceModelEvaluator(Evaluator):
    def __init__(self,
                 problem_model,
                 edge_count_weight = 1.0,
                 circularity_weight = 1.0,
                 not_missing_containment_weight = 1.0,
                 not_miscontained_weight = 1.0,
                 no_single_separation_weight = 1.0,
                #  spatial_dispersion_weight,
                 min_distance_weight = 1.0,
                 nodes_at_min_distance_weight = 1.0,
                 area_proportionality_weight = 1.0):
        self.edge_count_weight              = edge_count_weight
        self.circularity_weight             = circularity_weight
        self.not_missing_containment_weight = not_missing_containment_weight
        self.not_miscontained_weight        = not_miscontained_weight
        self.no_single_separation_weight    = no_single_separation_weight
        # self.spatial_dispersion_weight      = spatial_dispersion_weight
        self.min_distance_weight            = min_distance_weight
        self.nodes_at_min_distance_weight   = nodes_at_min_distance_weight
        self.area_proportionality_weight    = area_proportionality_weight
        self.problem = problem_model

    # best [0,1] worst
    def _calculate_nodes_not_missing_ratio(self): # O(m)
        # node_num = self.problem.hypergraph.shape[0]
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
    def _calculate_miscontained_nodes_measure(self, segment_vertex_ids, convex_hull, all_positions, r): # O(S*((n-d)*d + d*log(d))) <= O(n^3)
        def calculate_two_point_rectangle(start, end, r):
            #v = np.array([end[0]-start[0], end[1]-start[1]], dtype=np.float32)
            n = np.array([start[1]-end[1], end[0]-start[0]], dtype=np.float32)
            n /= np.linalg.norm(n)

            c1 = (start + r*n)
            c2 = (start - r*n)
            c3 = (end - r*n)
            c4 = (end + r*n)
            return np.vstack([c1, c2, c3, c4]).astype(np.float32)
        
        def calculate_two_point_hull_distance(start, end, r, point):
            rectangle_corners = calculate_two_point_rectangle(start, end, r)
            start_1, start_2, end_2, end_1 = rectangle_corners
            line_dist_1, _nearest_1 = self._get_line_segment_point_distance(start_1, end_1, point)
            line_dist_2, _nearest_2 = self._get_line_segment_point_distance(start_2, end_2, point)
            if line_dist_1 <= r and line_dist_2 <= r:
                return np.min(line_dist_1, line_dist_2)
            start_distance = np.linalg.norm(start - point)
            if start_distance <= r:
                return start_distance
            end_distance = np.linalg.norm(end - point)
            if end_distance <= r:
                return end_distance
            min_dist = min(line_dist_1, line_dist_2, start_distance, end_distance)
            if np.abs(min_dist) <= 1e-8:
                return 0.0
            return -1 * min_dist
    
        miscontained_vertex_num = 0
        #segment_vertex_num = 0
        max_possible_distance_to_polygon = np.linalg.norm(np.array(self.problem.size) / 2.0) # TODO
        for vertex_id in range(all_positions.shape[0]):
            if vertex_id in segment_vertex_ids:
                #segment_vertex_num += 1
                continue

            vertex = all_positions[vertex_id]
            if convex_hull.shape[0] > 2:
                distance_to_polygon = cv2.pointPolygonTest(convex_hull.astype(np.float32), tuple(vertex), True)
                if distance_to_polygon >= 0:
                    miscontained_vertex_num += distance_to_polygon / max_possible_distance_to_polygon
            elif convex_hull.shape[0] == 2:
                start, end = convex_hull
                distance_to_polygon = calculate_two_point_hull_distance(start, end, r, vertex)
                if distance_to_polygon >= 0:
                    miscontained_vertex_num += distance_to_polygon / max_possible_distance_to_polygon
            else:
                if np.linalg.norm(convex_hull[0] - vertex) < r:
                    miscontained_vertex_num += np.linalg.norm(r - convex_hull[0]) / max_possible_distance_to_polygon
        
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

    # def _measure_spatial_dispersion(self): # O(n)
    #     # test if it is from a uniform distribution
    #     # check if the sum of x & y differences to the center are 0

    # best [0,1] worst
    def _calculate_area_proportionality(self, edge_id, edge_hulls, r): # O(S) <= O(n)
        edge_area = sum([self._get_area(convex_hull, r) for convex_hull in edge_hulls])
        canvas_size = self.problem.size[0] * self.problem.size[1]
        all_vertex_num = self.problem.hypergraph.shape[0]
        edge_vertex_num = self.problem.hypergraph[:, edge_id].sum()
        return abs(edge_area / canvas_size - edge_vertex_num / all_vertex_num) # abs([0,1] - [0,1])

    # [0,sqrt(width^2+height^2)], [0, n]
    def _calculate_min_node_distance_and_occurences(self, positions): # O(n^2), this measure is global
        distances = pdist(positions)
        min_distance = distances.min()
        epsilon = 1e-8
        num_mins = (distances <= min_distance + epsilon).sum()
        return min_distance, num_mins

    # best [0,1] worst
    def _calculate_min_node_distance_to_target_ratio(self, min_distance_realisation): # O(n^2), this measure is global
        return 1.0 - min(min_distance_realisation / self.problem.min_node_distance, 1.0)

    # https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    def _get_line_segment_point_distance(self, start, end, point):
            line = start - end
            point_vector = (start - point)
            line_len = np.linalg.norm(line)
            line_unit_vector = line / line_len
            scaled_point_vector = point_vector * 1.0/line_len
            t = np.dot(line_unit_vector, scaled_point_vector)    
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            nearest = line * t
            dist = np.linalg.norm(nearest - point_vector)
            nearest = nearest + start
            return (dist, nearest)

#     def _do_line_segments_intersect(self, segment_1_endpoints, segment_2_endpoints):
#         p, p_plus_r = segment_1_endpoints
#         r = p_plus_r - p
#         q, q_plus_s = segment_2_endpoints
#         s = q_plus_s - q
        
#         r_cross_s = np.cross(r, s)
#         q_minus_p_cross_r = np.cross(q - p, r)

#         epsilon = 1e-8
#         if np.abs(r_cross_s) < epsilon:
#             if np.abs(q_minus_p_cross_r) < epsilon:
#                 return True # collinear
#             else:
#                 return False # parallel
        
#         t = np.cross(q-p, s) / r_cross_s
#         u = q_minus_p_cross_r / r_cross_s

#         if 0 <= t <= 1 and 0 <= u <= 1:
#             return True
#         else:
#             return False

#     # def _get_non_intersecting_edge_pairs(self):
#     #     h = self.problem.hypergraph
#     #     intersection_table = h.transpose() @ h
#     #     non_intersecting = np.where(~intersection_table)
#     #     symmetricity_mask = non_intersecting[0] <= non_intersecting[1]
#     #     non_intersecting = np.vstack((non_intersecting[0][symmetricity_mask], non_intersecting[1][symmetricity_mask]))
#     #     return non_intersecting

#     def _get_edge_intersection_table(self):
#         h = self.problem.hypergraph
#         intersection_table = h.transpose() @ h
#         return intersection_table
    
#     def _do_edge_components_intersect(self, all_positions, segment_1, segment_2):
#         # r = self._get_r()
#         # if len(segment_1) == 1:
#         #     for node_id in segment_2:
#         #         if np.linalg.norm(all_positions[node_id] - segment_1[0]) <= r:
#         #             return True
#         # if len(segment_2) == 1:
#         #     for node_id in segment_1:
#         #         if np.linalg.norm(all_positions[node_id] - segment_2[0]) <= r:
#         #             return True
#         # if len(segment_1) == 2:
#         #     start_1, end_1 = segment_1

    def _calculate_intersection_measure(self, all_positions, edge_components):
        edge_intersections = self._get_edge_intersection_table()
        all_segment_num = sum(map(len, edge_components))
        all_possible_intersections = all_segment_num * (all_segment_num - 1) / 2.0
        measure = 0.0
        for edge_1_id in range(len(edge_components) - 1):
            for edge_2_id in range(edge_1_id, len(edge_components)):
                is_intersecting = False
                if edge_intersections[edge_1_id, edge_2_id]:
                    continue
                for segment_1 in edge_components[edge_1_id]:
                    for segment_2 in edge_components[edge_2_id]:
                        is_intersecting = self._do_edge_components_intersect(all_positions, segment_1, segment_2)
                        if is_intersecting:
                            measure += 1.0 / all_possible_intersections


    def _get_r(self):
        r = self.problem.min_node_distance / 2.0
        return r

    def get_edgewise_and_global_scores(self, x):
        all_edge_components = self.problem.get_edge_components(x)
        edgewise_scores = np.full(len(all_edge_components), 1.0)
        all_positions = self.problem.extract_positions2D(x)

        r = self._get_r()

        for edge_id in range(len(all_edge_components)):
            segment_hulls = [self._get_convex_hull(segment, all_positions) for segment in all_edge_components[edge_id]]
            edge = self.problem.hypergraph[:, edge_id]
            segment_vertex_ids = np.where(edge)[0]

            nodes_not_missing_measure = self._calculate_nodes_not_missing_ratio()
            circularity_measure = sum([self._calculate_circularity_measure(segment_hull, r) / len(segment_hulls) for segment_hull in segment_hulls])
            miscounted_nodes_measure = sum([self._calculate_miscontained_nodes_measure(segment_vertex_ids, segment_hull, all_positions, r) / len(segment_hulls) for segment_hull in segment_hulls])
            edge_segment_num_measure = self._calculate_edge_segments_num_ratio(all_edge_components[edge_id])
            area_proportionality_measure = self._calculate_area_proportionality(edge_id, segment_hulls, r)
            single_node_separation_ratio = self._calculate_single_node_separations_ratio(all_edge_components[edge_id])

            edge_score  = self.edge_count_weight  * edge_segment_num_measure
            edge_score += self.circularity_weight * circularity_measure
            edge_score += self.not_miscontained_weight * miscounted_nodes_measure
            edge_score += self.not_missing_containment_weight * nodes_not_missing_measure
            edge_score += self.no_single_separation_weight * single_node_separation_ratio
            edge_score += self.area_proportionality_weight * area_proportionality_measure

            edgewise_scores[edge_id] = edge_score


        min_node_distance, min_distance_occurence_num = self._calculate_min_node_distance_and_occurences(all_positions)
        min_distance_measure = self._calculate_min_node_distance_to_target_ratio(min_node_distance)

        global_score = edgewise_scores.mean()
        global_score += self.min_distance_weight * min_distance_measure * min_distance_occurence_num

        return edgewise_scores, global_score

    def __call__(self, x):
        _edgewise_scores, global_score = self.get_edgewise_and_global_scores(x)
        return global_score
