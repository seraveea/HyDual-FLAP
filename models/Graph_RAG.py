import pandas as pd
from collections import Counter
from .noGraph_RAG import vanilla_RAG
import uuid
import networkx as nx
import numpy as np
import time
import sys
sys.path.insert(0, sys.path[0]+"/../")
from scripts.utils import generate_prompt_ll3


class graph_rag(vanilla_RAG):
    def __init__(self, client, time_series, summary, maper, embeder, preload_doc_path, style, language):
        super().__init__(client, time_series, summary, maper, embeder, preload_doc_path, style, language)
        self.graph = None
        self.stat_dict = None

    def graph_generating(self, symbol, data, static_data):
        """
        :param symbol: the center symbol
        :param data: dataframe with all tuples
        :param static_data: dataframe with business summary.
        :return: two dataframes, one with all events about the center symbol,
                                 one with all tuples in a 2-hop subgraph of center node
        this function first split center-related event and tuples, then build a graph
        extract the 2-hop subgraph, return all tuples in the subgraph
        """
        center = self.maper[symbol]  # 中心节点
        data = data[data['refined_entities_len'] < 6]

        tkg = nx.MultiGraph()
        for index, row in static_data.iterrows():
            tkg.add_edge(row['Subject'], row['Object'], label=row['uuid'])
        for index, row in data.iterrows():
            pairs = self.create_pairs(row['refined_entities'])
            for pair in pairs:
                tkg.add_edge(pair[0], pair[1], label=row['uuid'])
        path = nx.single_source_shortest_path(tkg, center, 2)
        neighbor_node = list(path.keys())
        sub_tkg = tkg.subgraph(neighbor_node)
        edge_data = [t[2]['label'] for t in list(sub_tkg.edges.data())]
        sub_data = data.merge(pd.DataFrame({'uuid': edge_data}), on=['uuid'])
        self.graph = tkg
        return sub_data

    def category_tuple_event(self, row, center):
        if self.stat_dict[row['Subject']] == 1 or self.stat_dict[row['Object']] == 1:
            # first it is an event
            if row['Subject'] == center or row['Object'] == center:
                # second it is about center node
                return 'Center Event'
            else:
                # it's not about the center node then mark it as tuple
                return 'Tuple'
        elif row['Subject'] == center and row['Object'] == center:
            # self-loop
            return 'Center Event'
        else:
            return 'Tuple'

    @staticmethod
    def l2_distance(x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def create_pairs(entities):
        if len(entities) == 1:
            return [(entities[0], entities[0])]
        else:
            return [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i + 1, len(entities))]


