from .noGraph_RAG import vanilla_RAG
import numpy as np
import sys
sys.path.insert(0, sys.path[0]+"/../")

"""
Notes: Graph RAG agent
"""


class graph_rag(vanilla_RAG):
    def __init__(self, client, time_series, summary, maper, embeder, preload_doc_path, style):
        super().__init__(client, time_series, summary, maper, embeder, preload_doc_path, style)
        self.graph = None
        self.stat_dict = None

    @staticmethod
    def l2_distance(x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def create_pairs(entities):
        if len(entities) == 1:
            return [(entities[0], entities[0])]
        else:
            return [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i + 1, len(entities))]


