# coding: utf-8
from hashlib import md5


class HashTool:
    @staticmethod
    def md5_hash(s: str) -> str:
        m = md5()
        m.update(s.encode('utf-8'))
        h = m.hexdigest()
        return str(h)
