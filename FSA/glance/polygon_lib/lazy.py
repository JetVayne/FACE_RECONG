# coding: utf-8
from shapely.geometry import Polygon


class LazyMethod:
    @staticmethod
    def get_polygon_len(coords: list):
        return Polygon(coords).length

    @staticmethod
    def get_polygon_area(coords: list) -> float:
        return Polygon(coords).area
