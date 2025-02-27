import cv2
import numpy as np

class HypergraphDrawer:
    def __init__(self, problem, solution):
        self._problem = problem
        self._solution = solution
        self._edge_components = problem.get_edge_components(solution)
    
    def _generate_colors(self, n):
        return (np.random.random((n,3)) * 255).astype(np.uint8)
    
    def _generate_empty_image(self, size, color=(255,255,255)):
        width  = size[0]
        height = size[1]
        img = np.full((height, width, 3), color, np.uint8)
        return img

    def _get_convex_hull(self, edge_segment, all_positions):
        mask = list(edge_segment)
        hull = cv2.convexHull(all_positions[mask].astype(np.float32))
        hull = hull.reshape((hull.shape[0], 2))
        return hull
    
    def _get_radius(self):
        return self._problem.min_node_distance / 2.0

    def _draw_segment(self, edge_img, segment_hull, color, segment_num):
        def draw_two_point_rectangle(edge_img, start, end, r, color):
            n = np.array([start[1]-end[1], end[0]-start[0]], dtype=np.float32)
            n /= np.linalg.norm(n)

            c1 = (start + r*n)
            c2 = (start - r*n)
            c3 = (end - r*n)
            c4 = (end + r*n)
            pts = np.vstack([c1, c2, c3, c4]).astype(np.float32)

            cv2.fillPoly(edge_img, [pts.astype(np.int32)], color)

        color = tuple(map(int, color))
        assert(len(segment_hull) > 0)
        if len(segment_hull) == 1:
            point = segment_hull[0]
            r = self._get_radius()
            cv2.circle(edge_img, tuple(point), int(r), color, cv2.FILLED)
        elif len(segment_hull) == 2:
            start, end = segment_hull
            r = self._get_radius()
            cv2.circle(edge_img, tuple(start), int(r), color, cv2.FILLED)
            cv2.circle(edge_img, tuple(end), int(r), color, cv2.FILLED)

            draw_two_point_rectangle(edge_img, start, end, r, color)
        else:
            cv2.fillPoly(edge_img, [segment_hull.astype(np.int32)], color)
            #boundingRect = segment_hull[:,0].min(), segment_hull[:,1].min(), segment_hull[:,0].max(), segment_hull[:,1].max()
            #color = (0, 0, 0) 
            #thickness = 1 
            #edge_img = cv2.rectangle(edge_img, (boundingRect[0], boundingRect[1]), (boundingRect[2], boundingRect[3]), color, thickness) 

    def _draw_points(self, img, all_positions, color=(0,0,0), r=3):
        for point in all_positions:
            cv2.circle(img, tuple(point), r, color, cv2.FILLED)

    def build_image(self, colors=None):
        edge_num = len(self._edge_components)
        assert(colors is None or len(colors) == edge_num)

        if colors is None:
            colors = self._generate_colors(len(self._edge_components))
        img = self._generate_empty_image(self._problem.size)

        all_positions = np.round(self._problem.extract_positions2D(self._solution)).astype(np.int32)
        edge_components = self._problem.get_edge_components(self._solution)

        for edge_id in range(edge_num):
            color = colors[edge_id]
            edge_img = self._generate_empty_image(self._problem.size)
            for segment in edge_components[edge_id]:
                hull = self._get_convex_hull(segment, all_positions)
                self._draw_segment(edge_img, hull, color, len(edge_components[edge_id]))
            alpha = 1.0 / (edge_id + 1)
            img = cv2.addWeighted(edge_img, alpha, img, 1.0-alpha, gamma=0)


        self._draw_points(img, all_positions)

        return img

    def show(self, img=None, colors=None, wait_key=0, window_title='Window'):
        if img is None:
            img = self.build_image(colors)
        cv2.imshow(window_title, img)
        cv2.waitKey(wait_key)
