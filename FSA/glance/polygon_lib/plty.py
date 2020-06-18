# coding: utf-8
import numpy as np
from glance.jf_ult.geom_tool import GeomTool


class PltyMethod:
    @staticmethod
    def polygon_pt_sort(points):
        n = len(points)
        cx = float(sum(x for x, y in points)) / n
        cy = float(sum(y for x, y in points)) / n

        cornersWithAngles = []
        for x, y in points:
            an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
            cornersWithAngles.append((x, y, an))

        cornersWithAngles.sort(key=lambda tup: tup[2])
        points = [(c[0], c[1]) for c in cornersWithAngles]
        return points

    @staticmethod
    def get_perimeter(points: list) -> float:
        points = PltyMethod.polygon_pt_sort(points)
        result = 0
        for index, point in enumerate(points):
            dest_index = index + 1
            if dest_index < len(points):
                dest_point = points[dest_index]

            else:
                dest_index = 0
                dest_point = points[dest_index]

            result += GeomTool.get_pt_dist(point, dest_point)

        return result

    @staticmethod
    def get_area(points: list):
        points = PltyMethod.polygon_pt_sort(points)
        n = len(points)
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        area = abs(area) / 2.0

        return area
