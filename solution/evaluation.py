import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import cv2

from solution.drawing import HypergraphDrawer

class Evaluator:
    def __call__(self, x):
        pass

class DistanceModelEvaluator(Evaluator):
    class Summary:
        def __init__(self):
            self.falsely_contained_node_num = 0
            self.missing_node_num = 0
            self.intersection_num = 0
            self.invalid_intersection_num = 0
            self.closer_than_min_distance_num = 0
            self.segment_num = 0
            self.single_node_segment_num = 0
        
        def __str__(self):
            return f'''
falsely_contained_node_num:   {self.falsely_contained_node_num}
missing_node_num:             {self.missing_node_num}
intersection_num:             {self.intersection_num}
invalid_intersection_num:     {self.invalid_intersection_num}
closer_than_min_distance_num: {self.closer_than_min_distance_num}
segment_num:                  {self.segment_num}
single_node_segment_num:      {self.single_node_segment_num}
'''

    def __init__(self,
                 edge_count_weight = 1.0,
                 circularity_weight = 1.0,
                 not_missing_containment_weight = 1.0,
                 not_miscontained_weight = 1.0,
                 edge_segment_size_weight = 1.0,
                 min_distance_weight = 1.0,
                 min_distance_occurence_weight = 1.0,
                 area_proportionality_weight = 1.0,
                 all_intersection_measure_weight = 1.0,
                 invalid_intersection_weight = 1.0,
                 no_single_separation_weight = 1.0,
                 properties_only=False,
                 debug=None):
        self.edge_count_weight               = edge_count_weight
        self.circularity_weight              = circularity_weight
        self.not_missing_containment_weight  = not_missing_containment_weight
        self.not_miscontained_weight         = not_miscontained_weight
        self.edge_segment_size_weight        = edge_segment_size_weight
        # self.spatial_dispersion_weight     = spatial_dispersion_weight
        self.min_distance_weight             = min_distance_weight
        self.min_distance_occurence_weight   = min_distance_occurence_weight
        self.area_proportionality_weight     = area_proportionality_weight
        self.all_intersection_measure_weight = all_intersection_measure_weight
        self.invalid_intersection_weight     = invalid_intersection_weight
        self.no_single_separation_weight     = no_single_separation_weight
        self.properties_only = properties_only
        self.debug = debug
        self.summary = self.Summary()

    def clone(self):
        return DistanceModelEvaluator(*self.dump_as_args())
    
    def dump_as_args(self):
        return (
            self.edge_count_weight,
            self.circularity_weight,
            self.not_missing_containment_weight,
            self.not_miscontained_weight,
            self.edge_segment_size_weight,
            self.min_distance_weight,
            self.min_distance_occurence_weight,
            self.area_proportionality_weight,
            self.all_intersection_measure_weight,
            self.invalid_intersection_weight,
            self.no_single_separation_weight,
            self.properties_only,
            self.debug)

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

    def _get_bounding_rect(self, problem, convex_hull, r):
        get_r_rectangle = lambda vertex_position: (max(vertex_position[0] - r, 0), max(vertex_position[1] - r, 0), min(vertex_position[0] + r, problem.size[0]), min(vertex_position[1] + r, problem.size[1]))
        if len(convex_hull) == 1:
            return get_r_rectangle(convex_hull[0])
        elif len(convex_hull) == 2:
            start_box = get_r_rectangle(convex_hull[0])
            end_box = get_r_rectangle(convex_hull[1])
            return min(start_box[0], end_box[0]), min(start_box[1], end_box[1]), max(start_box[2], end_box[2]), max(start_box[3], end_box[3])
        else:
            return convex_hull[:,0].min(), convex_hull[:,1].min(), convex_hull[:,0].max(), convex_hull[:,1].max()

    # best [0, 1) worst
    def _calculate_miscontained_nodes_measure(self, problem, segment_vertex_ids, convex_hull, all_positions, r): #O(S*((n-d)*log(d))) <= O(m*n*log(n))
        miscontained_vertex_num = 0
        miscontained_vertex_measure = 0
        #segment_vertex_num = 0
        #max_possible_distance_to_polygon = min(problem.size[0] / 2,  problem.size[1] / 2)
        bounding_rect = self._get_bounding_rect(problem, convex_hull, r)
        max_possible_distance_to_polygon = min(bounding_rect[2] - bounding_rect[0], bounding_rect[3] - bounding_rect[1]) / 2
        not_contained_vertex_num = (len(all_positions) - len(segment_vertex_ids))
        denominator = max_possible_distance_to_polygon * not_contained_vertex_num

        for vertex_id in range(all_positions.shape[0]):
            if vertex_id in segment_vertex_ids:
                #segment_vertex_num += 1
                continue

            vertex = all_positions[vertex_id]
            if convex_hull.shape[0] > 2:
                distance_to_polygon = cv2.pointPolygonTest(convex_hull.astype(np.float32), tuple(vertex), True)
                if distance_to_polygon >= 0:
                    miscontained_vertex_num += 1
                    miscontained_vertex_measure += distance_to_polygon / denominator
            elif convex_hull.shape[0] == 2:
                start, end = convex_hull
                distance_to_polygon, is_point_inside = self._calculate_two_point_hull_distance(start, end, r, vertex)
                if is_point_inside:
                    miscontained_vertex_num += 1
                    miscontained_vertex_measure += distance_to_polygon / denominator
            else:
                distance_to_polygon = np.linalg.norm(convex_hull[0] - vertex)
                if distance_to_polygon < r:
                    miscontained_vertex_num += 1
                    miscontained_vertex_measure += distance_to_polygon / denominator
        
        self.summary.falsely_contained_node_num += miscontained_vertex_num
        
        if self.properties_only:
            if miscontained_vertex_num > 0:
                return 1.0
            return 0.0
        return miscontained_vertex_measure

    def _calculate_two_point_rectangle_corners(self, start, end, r):
        #v = np.array([end[0]-start[0], end[1]-start[1]], dtype=np.float32)
        n = np.array([start[1]-end[1], end[0]-start[0]], dtype=np.float32)
        n /= np.linalg.norm(n)

        c1 = (start + r*n)
        c2 = (start - r*n)
        c3 = (end - r*n)
        c4 = (end + r*n)
        return np.vstack([c1, c2, c3, c4]).astype(np.float32)

    def _calculate_two_point_hull_distance(self, start, end, r, point): # returns distance, is_point_inside
        rectangle_corners = self._calculate_two_point_rectangle_corners(start, end, r)
        start_1, start_2, end_2, end_1 = rectangle_corners
        line_dist_1, _nearest_1 = self._get_line_segment_point_distance(start_1, end_1, point)
        line_dist_2, _nearest_2 = self._get_line_segment_point_distance(start_2, end_2, point)
        start_distance = np.linalg.norm(start - point)
        end_distance = np.linalg.norm(end - point)
        min_dist = min(line_dist_1, line_dist_2, start_distance, end_distance)
        if np.abs(min_dist) <= 1e-8:
            min_dist = 0.0
        if (line_dist_1 <= r and line_dist_2 <= 2*r) or (line_dist_1 <= 2*r and line_dist_2 <= r) or start_distance <= r or end_distance <= r:
            return min_dist, True
        return min_dist, False

    # best [0,1) worst
    def _calculate_edge_segments_num_ratio(self, edge_segments):
        #return len(edge_components) / sum(map(len, edge_components))
        return 1.0 - 1.0 / len(edge_segments)

    # best [0, 1] worst
    def _calculate_single_node_separations_ratio(self, edge_segments): # O(S) <= O(n)
        one_long_segment_num = sum([1 for segment in edge_segments if len(segment) == 1])
        if len(edge_segments) > 1:
            self.summary.single_node_segment_num = one_long_segment_num
            if self.properties_only and one_long_segment_num >= 1:
                return 1.0
            return one_long_segment_num / len(edge_segments)
        else:
            return 0.0
    
    def _calculate_segment_size_measure(self, problem, edge_segments, edge_id): # O(S) <= O(n)
        edge_vertex_num = problem.hypergraph[:, edge_id].sum()
        return sum([abs(len(segment) / edge_vertex_num - 1 / len(edge_segments)) / len(edge_segments) for segment in edge_segments])

    # def _measure_spatial_dispersion(self): # O(n)
    #     # test if it is from a uniform distribution
    #     # check if the sum of x & y differences to the center are 0

    # best [0,1] worst
    def _calculate_area_proportionality(self, problem, edge_id, edge_hulls, r): # O(S) <= O(n)
        #edge_area = sum([self._get_area(convex_hull, r) for convex_hull in edge_hulls])
        #canvas_size = problem.size[0] * problem.size[1]
        #all_vertex_num = problem.hypergraph.shape[0]
        #edge_vertex_num = problem.hypergraph[:, edge_id].sum()
        #return abs(edge_area / canvas_size - edge_vertex_num / all_vertex_num) # abs([0,1] - [0,1])
        
        segment_areas = [self._get_area(convex_hull, r) for convex_hull in edge_hulls]
        canvas_size = problem.size[0] * problem.size[1]
        all_vertex_num = problem.hypergraph.shape[0]
        measure = [abs(segment_areas[i] / canvas_size - len(edge_hulls[i]) / all_vertex_num) / len(edge_hulls) for i in range(len(segment_areas))]
        return sum(measure)

    # [0,sqrt(width^2+height^2)], [0, n]
    def _calculate_min_node_distance_and_below_target_occurences(self, problem, positions): # O(n^2), this measure is global
        distances = pdist(positions)
        min_distance = distances.min()
        #epsilon = 1e-8
        #num_mins = (distances <= min_distance + epsilon).sum()
        num_below_min = (distances < problem.min_node_distance).sum() # pdist only gives only one value for d(u,v), d(v,u)
        return min_distance, num_below_min

    # best [0,1] worst
    def _calculate_min_node_distance_to_target_ratio(self, problem, min_distance_realisation): # O(n^2), this measure is global
        if min_distance_realisation < 1e-8:
            return 1.0
        distance_to_min_threshold_ratio = min(min_distance_realisation / problem.min_node_distance, 1.0)
        if self.properties_only and distance_to_min_threshold_ratio < 1.0:
            return 1.0
        return 1.0 - distance_to_min_threshold_ratio

    # https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    def _get_line_segment_point_distance(self, start, end, point):
            epsilon = 1e-8
            if np.linalg.norm(start - end) < epsilon:
                start_diff = np.linalg.norm(start - point)
                end_diff = np.linalg.norm(start - point)
                return (start_diff, start) if start_diff < end_diff else (end_diff, end)
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

    def _do_line_segments_intersect(self, segment_1_endpoints, segment_2_endpoints):
        p, p_plus_r = segment_1_endpoints
        r = p_plus_r - p
        q, q_plus_s = segment_2_endpoints
        s = q_plus_s - q
        
        r_cross_s = np.cross(r, s)
        q_minus_p_cross_r = np.cross(q - p, r)

        epsilon = 1e-8
        if np.abs(r_cross_s) < epsilon:
            if np.abs(q_minus_p_cross_r) < epsilon:
                return True # collinear
            else:
                return False # parallel
        
        t = np.cross(q-p, s) / r_cross_s
        u = q_minus_p_cross_r / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
        else:
            return False

    # best (yes) [0, 1] worst (no) (by nodes)
    # if the given node has no neighbours at all then any neighbour is accepted with the best (0) value
    def _is_closest_neighbour_edge_neighbour(self, problem, all_positions): # O(n^3)
        distances = distance_matrix(all_positions, all_positions, threshold=10**8)
        np.fill_diagonal(distances, np.inf)
        closest_neighbour_node_id = np.apply_along_axis(np.argmin, 1, distances)
        closest_neighbour_node_id_pairs = np.stack((np.arange(closest_neighbour_node_id.size), closest_neighbour_node_id), axis=1)

        intersections = self._get_node_intersection_table(problem)
        has_neighbour = (intersections.sum(axis=1) > 1) # if it has at least one intersection then one of those is with itself (could also be that it isn't in any edges)

        is_edge_neighbour = lambda node_id_pair, intersections, has_neighbour : intersections[node_id_pair[0], node_id_pair[1]] or not has_neighbour[node_id_pair[0]]
        is_neighbour_point_vertex_neighbour = np.apply_along_axis(is_edge_neighbour, 1, closest_neighbour_node_id_pairs, intersections, has_neighbour).astype(np.float32)
        return is_neighbour_point_vertex_neighbour

    # best [0, 1] worst
    def _is_closest_neighbour_edge_neighbour_edge_measure(self, problem, edge_id, all_positions, is_closest_from_common_edge): # alternative heuristics to edge intersections O(m*n) + O(n^3)
        #is_closest_from_common_edge = self._is_closest_neighbour_edge_neighbour(problem, all_positions)
        edge = problem.hypergraph[:, edge_id]
        if edge.sum() == 0:
            return 0.0
        edge_vertex_ids = np.where(edge)[0]

        num_closest_from_commen_edge = is_closest_from_common_edge[edge_vertex_ids].sum()
        if self.properties_only:
            return 1.0 if num_closest_from_commen_edge < edge_vertex_ids.size else 0.0
        return 1.0 - num_closest_from_commen_edge / edge_vertex_ids.size
    
    def _get_node_intersection_table(self, problem):
        h = problem.hypergraph
        intersection_table = h @ h.transpose()
        return intersection_table

    def _get_edge_intersection_table(self, problem):
        h = problem.hypergraph
        intersection_table = h.transpose() @ h
        return intersection_table
    
    def _do_edge_components_intersect(self, problem, all_positions, segment_1, segment_2):
        def do_1_1_len_intersect(all_positions, segment_1_list, segment_2_list, r):
            point_1 = all_positions[segment_1_list[0]]
            point_2 = all_positions[segment_2_list[0]]
            if np.linalg.norm(point_1 - point_2) <= r:
                return True
            return False
        def do_1_2_len_intersect(all_positions, len_1_segment_list, len_2_segment_list, r):
            point_1 = all_positions[len_1_segment_list[0]]
            start_id, end_id = len_2_segment_list
            start = all_positions[start_id]
            end = all_positions[end_id]
            if np.linalg.norm(start - end) < 1e-8:
                return do_1_1_len_intersect(all_positions, len_1_segment_list, len_2_segment_list, r)
            if self._calculate_two_point_hull_distance(start, end, r, point_1)[0] <= r:
                return True
            return False
        def do_1_3_len_intersect(all_positions, len_1_segment_list, len_3_segment_list, r): # !!! in this case we don't check if two (or more) vertices are in the same position or not
            point_1 = all_positions[len_1_segment_list[0]]
            hull_2_line_endpoint_ids = [[len_3_segment_list[i], len_3_segment_list[(i+1) % len(len_3_segment_list)]] for i in range(len(len_3_segment_list))]
            for start_id, end_id in hull_2_line_endpoint_ids:
                start = all_positions[start_id]
                end = all_positions[end_id]
                if self._get_line_segment_point_distance(start, end, point_1)[0] <= r:
                    return True
            return False
        def do_2_2_len_intersect(all_positions, segment_1_list, segment_2_list, r):
            start_1, end_1 = all_positions[segment_1_list]
            start_2, end_2 = all_positions[segment_2_list]
            if np.linalg.norm(start_1 - end_1) < 1e-8:
                return do_1_2_len_intersect(all_positions, segment_1_list, segment_2_list, r) # recursively checks if segment_2 is in fact 1 long or not
            elif np.linalg.norm(start_2 - end_2) < 1e-8:
                return do_1_2_len_intersect(all_positions, segment_2_list, segment_1_list, r)
            if (np.linalg.norm(start_1 - start_2) <= r or
                np.linalg.norm(start_1 - end_2) <= r or
                np.linalg.norm(end_1 - start_2) <= r or
                np.linalg.norm(end_1 - end_2) <= r):
                return True
            rectangle_corners = self._calculate_two_point_rectangle_corners(start_1, end_1, r)
            line_start_11, line_start_12, line_end_12, line_end_11 = rectangle_corners
            if (self._get_line_segment_point_distance(line_start_11, line_end_11, start_2)[0] <= r or
                self._get_line_segment_point_distance(line_start_11, line_end_11, end_2)[0] <= r or
                self._get_line_segment_point_distance(line_start_12, line_end_12, start_2)[0] <= r or
                self._get_line_segment_point_distance(line_start_12, line_end_12, end_2)[0] <= r):
                return True
            rectangle_corners = self._calculate_two_point_rectangle_corners(start_2, end_2, r)
            line_start_21, line_start_22, line_end_22, line_end_21 = rectangle_corners
            if (self._get_line_segment_point_distance(line_start_21, line_end_21, start_1)[0] <= r or
                self._get_line_segment_point_distance(line_start_21, line_end_21, end_1)[0] <= r or
                self._get_line_segment_point_distance(line_start_22, line_end_22, start_1)[0] <= r or
                self._get_line_segment_point_distance(line_start_22, line_end_22, end_1)[0] <= r):
                return True
            endpoints_11 = np.array([line_start_11, line_end_11], dtype=np.float32)
            endpoints_12 = np.array([line_start_12, line_end_12], dtype=np.float32)
            endpoints_21 = np.array([line_start_21, line_end_21], dtype=np.float32)
            endpoints_22 = np.array([line_start_22, line_end_22], dtype=np.float32)
            if (self._do_line_segments_intersect(endpoints_11, endpoints_21) or
                self._do_line_segments_intersect(endpoints_11, endpoints_22) or
                self._do_line_segments_intersect(endpoints_12, endpoints_21) or
                self._do_line_segments_intersect(endpoints_12, endpoints_22)):
                return True
            return False
        def do_2_3_len_intersect(all_positions, len_2_segment_list, len_3_segment_list, r): # !!! in this case we don't check the second list has points in the same position
            start_1, end_1 = all_positions[len_2_segment_list]
            if np.linalg.norm(start_1 - end_1) < 1e-8:
                return do_1_3_len_intersect(all_positions, len_2_segment_list, len_3_segment_list, r)
            hull_2_line_endpoint_ids = [[len_3_segment_list[i], len_3_segment_list[(i+1) % len(len_3_segment_list)]] for i in range(len(len_3_segment_list))]
            rectangle_corners = self._calculate_two_point_rectangle_corners(start_1, end_1, r)
            line_start_1, line_start_2, line_end_2, line_end_1 = rectangle_corners
            endpoints_1a = np.array([line_start_1, line_end_1], dtype=np.float32)
            endpoints_1b = np.array([line_start_2, line_end_2], dtype=np.float32)
            for endpoint_ids in hull_2_line_endpoint_ids:
                endpoints_2 = all_positions[endpoint_ids]
                if (self._get_line_segment_point_distance(endpoints_2[0], endpoints_2[1], start_1)[0] <= r or
                    self._get_line_segment_point_distance(endpoints_2[0], endpoints_2[1], end_1)[0] <= r):
                    return True
                if self._do_line_segments_intersect(endpoints_1a, endpoints_2) or self._do_line_segments_intersect(endpoints_1b, endpoints_2):
                    return True
            return False

        def do_3_3_len_intersect(all_positions, segment_1_list, segment_2_list): # !!! in this case we don't check if two (or more) vertices are in the same position or not
            hull_1_line_endpoint_ids = [[segment_1_list[i], segment_1_list[(i+1) % len(segment_1_list)]] for i in range(len(segment_1_list))]
            hull_2_line_endpoint_ids = [[segment_2_list[i], segment_2_list[(i+1) % len(segment_2_list)]] for i in range(len(segment_2_list))]
            for endpoint_ids_1 in hull_1_line_endpoint_ids:
                for endpoint_ids_2 in hull_2_line_endpoint_ids:
                    endpoints_1 = all_positions[endpoint_ids_1]
                    endpoints_2 = all_positions[endpoint_ids_2]
                    if self._do_line_segments_intersect(endpoints_1, endpoints_2):
                        return True
            return False

        r = self._get_r(problem)
        segment_1_list = list(segment_1)
        segment_2_list = list(segment_2)
        if len(segment_1) == 1:
            if len(segment_2) == 1:
                return do_1_1_len_intersect(all_positions, segment_1_list, segment_2_list, r)
            elif len(segment_2) == 2:
                return do_1_2_len_intersect(all_positions, segment_1_list, segment_2_list, r)
            else:
                return do_1_3_len_intersect(all_positions, segment_1_list, segment_2_list, r)
        elif len(segment_1) == 2:
            if len(segment_2) == 1:
                return do_1_2_len_intersect(all_positions, segment_2_list, segment_1_list, r)
            elif len(segment_2) == 2:
                return do_2_2_len_intersect(all_positions, segment_1_list, segment_2_list, r)
            else:
                return do_2_3_len_intersect(all_positions, segment_1_list, segment_2_list, r)
        else:
            if len(segment_2) == 1:
                return do_1_3_len_intersect(all_positions, segment_2_list, segment_1_list, r)
            elif len(segment_2) == 2:
                return do_2_3_len_intersect(all_positions, segment_2_list, segment_1_list, r)
            else:
                return do_3_3_len_intersect(all_positions, segment_1_list, segment_2_list)

    def _calculate_intersection_measure(self, problem, all_positions, edge_components):
        nC2 = lambda n: n * (n - 1) / 2.0
        edge_intersections = self._get_edge_intersection_table(problem)
        all_segment_num = sum(map(len, edge_components))
        all_possible_intersections = nC2(all_segment_num)
        all_possible_non_neighbour_intersections = all_possible_intersections
        intersection_measure = 0.0
        non_neighbour_intersection_measure = 0.0

        #distances = distance_matrix(all_positions, all_positions)
        #distances += np.diag(np.full(len(all_positions), np.inf))
        #has_same_position = np.any(distances <= 1e-8)
#
        #if has_same_position:
        #    return 1

        for edge_1_id in range(len(edge_components) - 1):
            all_possible_non_neighbour_intersections -= nC2(len(edge_components[edge_1_id]))
            for edge_2_id in range(edge_1_id + 1, len(edge_components)):
                is_neighbour = False
                if edge_intersections[edge_1_id, edge_2_id]:
                    all_possible_non_neighbour_intersections -= (len(edge_components[edge_1_id]) * len(edge_components[edge_2_id]))
                    is_neighbour = True
                for segment_1 in edge_components[edge_1_id]:
                    for segment_2 in edge_components[edge_2_id]:
                        is_intersecting = self._do_edge_components_intersect(problem, all_positions, segment_1, segment_2)
                        if is_intersecting:
                            intersection_measure += 1.0
                            if not is_neighbour:
                                non_neighbour_intersection_measure += 1.0
        all_possible_intersections -= nC2(len(edge_components[-1])) # the iteration finished before the last item

        self.summary.intersection_num += intersection_measure
        self.summary.invalid_intersection_num += non_neighbour_intersection_measure

        if self.properties_only:
            return intersection_measure / all_possible_intersections, non_neighbour_intersection_measure

        return intersection_measure / all_possible_intersections, non_neighbour_intersection_measure / max(all_possible_non_neighbour_intersections, 1)

    def _get_r(self, problem):
        r = problem.min_node_distance / 2.0
        return r

    def get_edgewise_and_global_scores(self, problem, x):
        self.summary = self.Summary()
        all_edge_components = problem.get_edge_components(x)
        edgewise_scores = np.full(len(all_edge_components), 1.0)
        all_positions = problem.extract_positions2D(x)
        is_closest_from_common_edge = self._is_closest_neighbour_edge_neighbour(problem, all_positions)

        r = self._get_r(problem)
        for edge_id in range(len(all_edge_components)):
            segment_hulls = [self._get_convex_hull(segment, all_positions) for segment in all_edge_components[edge_id]]
            edge = problem.hypergraph[:, edge_id]
            segment_vertex_ids = np.where(edge)[0]

            nodes_not_missing_measure = self._calculate_nodes_not_missing_ratio()
            circularity_measure = sum([self._calculate_circularity_measure(segment_hull, r) / len(segment_hulls) for segment_hull in segment_hulls])
            miscounted_nodes_measure = sum([self._calculate_miscontained_nodes_measure(problem, segment_vertex_ids, segment_hull, all_positions, r) / len(segment_hulls) for segment_hull in segment_hulls])
            edge_segment_num_measure = self._calculate_edge_segments_num_ratio(all_edge_components[edge_id])
            area_proportionality_measure = self._calculate_area_proportionality(problem, edge_id, segment_hulls, r)
            single_node_separation_ratio = self._calculate_single_node_separations_ratio(all_edge_components[edge_id])
            edge_segment_size_measure = self._calculate_segment_size_measure(problem, all_edge_components[edge_id], edge_id)
            closest_neighbour_measure = self._is_closest_neighbour_edge_neighbour_edge_measure(problem, edge_id, all_positions, is_closest_from_common_edge)

            edge_score  = self.edge_count_weight  * edge_segment_num_measure
            edge_score += self.circularity_weight * circularity_measure
            edge_score += self.not_miscontained_weight * miscounted_nodes_measure
            edge_score += self.not_missing_containment_weight * nodes_not_missing_measure
            edge_score += self.no_single_separation_weight * single_node_separation_ratio
            edge_score += self.edge_segment_size_weight * edge_segment_size_measure
            edge_score += self.area_proportionality_weight * area_proportionality_measure
            edge_score /= len(all_edge_components)
            #edge_score += self.all_intersection_measure_weight * closest_neighbour_measure

            edgewise_scores[edge_id] = edge_score

        min_node_distance, below_min_distance_num = self._calculate_min_node_distance_and_below_target_occurences(problem, all_positions)
        self.summary.closer_than_min_distance_num = below_min_distance_num
        self.summary.segment_num = sum(map(len, all_edge_components))
        min_distance_measure = self._calculate_min_node_distance_to_target_ratio(problem, min_node_distance)

        all_intersection_measure, non_neighbour_intersection_measure = self._calculate_intersection_measure(problem, all_positions, all_edge_components)

        global_score = edgewise_scores.mean()
        #global_score = edgewise_scores.max()
        global_score += self.min_distance_weight * min_distance_measure
        global_score += self.min_distance_occurence_weight * below_min_distance_num / (len(all_positions) * (len(all_positions) - 1) / 2)
        global_score += self.all_intersection_measure_weight * all_intersection_measure
        global_score += self.invalid_intersection_weight * non_neighbour_intersection_measure

        return edgewise_scores, global_score

    def __call__(self, x, problem, return_edgwise_scores=False): # order is needed for apply_along_axis
        x_local = x.copy()
        x_local = np.round(x_local, 6)

        _edgewise_scores, global_score = self.get_edgewise_and_global_scores(problem, x_local)

        if self.debug is not None:
            drawer = HypergraphDrawer(problem, x_local)
            drawer.show(wait_key=self.debug)

            print(global_score)

        if return_edgwise_scores:
            return np.append(global_score, _edgewise_scores)
        else:
            return global_score
    
    def get_summary(self, x, problem):
        self.summary = self.Summary()
        self.__call__(x, problem, False)
        value = self.summary
        self.summary = self.Summary()
        return value

class HGCNEvaluator(DistanceModelEvaluator):
    def __call__(self, x, problem, neural_network, laplacians):
        score_sum = 0
        batch_size = len(problem.hypergraphs)
        for i in range(batch_size):
            subproblem = problem.clone(False)
            subproblem.hypergraph = problem.hypergraphs[i]
            temp_network = neural_network.clone(False)
            temp_network.load_row_vector(x)
            output = temp_network.predict(subproblem, laplacians[i])
            score = super().__call__(output.flatten(order='F'), subproblem, False)
            score_sum += score
        return score_sum / batch_size
