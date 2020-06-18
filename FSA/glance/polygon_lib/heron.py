# coding: utf-8
import math
import numpy as np
from glance.jf_ult.geom_tool import GeomTool


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.arr = [x, y]


class HeronMethod:
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
        points = HeronMethod.polygon_pt_sort(points)
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
    def get_area(points):
        points = HeronMethod.polygon_pt_sort(points)
        points = [Point(p[0], p[1]) for p in points]

        # 計算多邊形面積
        area = 0
        if len(points) < 3:
            raise Exception("error")

        p1 = points[0]
        for i in range(1, len(points) - 1):
            p2 = points[i]
            p3 = points[i + 1]

            # 計算向量
            vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
            vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)

            # 判斷順時針還是逆時針，順時針面積為正，逆時針面積為負
            vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x  # 判斷正負方向比較有意思
            sign = 0
            if vecMult > 0:
                sign = 1

            elif vecMult < 0:
                sign = -1

            triArea = HeronMethod.GetAreaOfTriangle(p1, p2, p3) * sign
            area += triArea
        return abs(area)

    @staticmethod
    def GetAreaOfTriangle(p1, p2, p3):
        area = 0
        p1p2 = GeomTool.get_pt_dist(p1.arr, p2.arr)
        p2p3 = GeomTool.get_pt_dist(p2.arr, p3.arr)
        p3p1 = GeomTool.get_pt_dist(p3.arr, p1.arr)
        s = (p1p2 + p2p3 + p3p1) / 2
        area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)  # 海倫公式
        area = math.sqrt(area)
        return area


face_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]  # bad
face_coords = HeronMethod.polygon_pt_sort(face_coords)

perm = HeronMethod.get_perimeter(face_coords)
area = HeronMethod.get_area(face_coords)

print(perm)
print('[海倫法] ', '面積: {}'.format(area), '結果: {}'.format(perm / area))
