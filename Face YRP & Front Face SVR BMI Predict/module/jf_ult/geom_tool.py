# coding: utf-8
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
from area import area


class GeomTool:
    @staticmethod
    def get_line_intersect(line1: tuple, line2: tuple) -> (int, int):
        x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(x_diff, y_diff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, x_diff) / div
        y = det(d, y_diff) / div
        return int(x), int(y)

    @staticmethod
    def get_mid_point(point1: tuple, point2: tuple) -> (int, int):
        x = (point1[0] + point2[0]) // 2
        y = (point1[1] + point2[1]) // 2
        return x, y

    @staticmethod
    def get_pt_dist(p1: tuple, p2: tuple):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def get_prj_point(point: tuple, line: tuple):
        point = Point(point)
        line = LineString(line)
        x = np.array(point.coords[0])

        u = np.array(line.coords[0])
        v = np.array(line.coords[len(line.coords) - 1])

        n = v - u
        n /= np.linalg.norm(n, 2)

        p = u + n * np.dot(x - u, n)

        return int(p[0]), int(p[1])

    @staticmethod
    def get_polygon_len(coords: list):
        return Polygon(coords).length

    @staticmethod
    def get_polygon_area(coords: list) -> float:
        return Polygon(coords).area
