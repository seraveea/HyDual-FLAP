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

"""
Notes: Graph RAG agent
"""


class graph_rag(vanilla_RAG):
    def __init__(self, client, time_series, summary, maper, embeder, preload_doc_path, style):
        super().__init__(client, time_series, summary, maper, embeder, preload_doc_path, style)
        self.graph = None
        self.stat_dict = None

    def graph_rag_reply(self, symbol, date, sub_dataset, static_data, topk, backbone, reserve_for_event=5):
        """
        :return: one df with llm reply, date, symbol and ground truth, one df with retrieval summary files
        """
        # **********
        start_time = time.time()
        # **********
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                docs = sub_dataset.filter(lambda e: self.retrieve_based_on_given_list(e, doc_list))
                separate_dict = sub_dataset[sub_dataset['url'].isin(doc_list)].to_dict(orient='records')
            except:
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                     'N/A',
                                     [{'url': 'empty'}],
                                     0)
        else:
            sub_dataset['uuid'] = sub_dataset['published time'].apply(lambda x: uuid.uuid4())  # dynamic sentences
            static_data['uuid'] = static_data['published time'].apply(lambda x: uuid.uuid4())  # static sentences
            target_data = self.graph_generating(symbol, sub_dataset, static_data)
            # target_data stored all tuples in 2-hop subgraph of center node

            query = self.default_query(symbol, date)
            symbol_embedding = np.array(self.embeder.embed(query))
            # symbol embedding is the "query vector", is an averaged vector of all recent events
            target_data['l2_distance'] = target_data['embedding'].apply(lambda x: self.l2_distance(x, symbol_embedding))
            sorted_document = target_data.sort_values(by='l2_distance')  # most related documents are in the head

            if sorted_document.shape[0] < topk:
                separate_dict = sorted_document.to_dict(orient='records')
            else:
                separate_dict = sorted_document.head(topk).to_dict(orient='records')  # use closet documents

        time_series, ground_truth = self.find_ground_truth(symbol, date)
        retrieval_prompt = generate_prompt_ll3(separate_dict, date, symbol, time_series)
        # **********
        end_time = time.time()
        run_time = end_time - start_time
        self.run_times.append(run_time)
        # **********
        if self.client == 'no model':
            return self.only_return_retrieved_files(separate_dict, symbol, date, ground_truth)
        elif self.client == 'gpt4o':
            return self.gpt_reply(retrieval_prompt, symbol, date, ground_truth, separate_dict)
        else:
            return self.client_reply(retrieval_prompt, symbol, date, ground_truth, separate_dict)

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
        return np.linalg.norm(x - y)

    @staticmethod
    def create_pairs(entities):
        if len(entities) == 1:
            return [(entities[0], entities[0])]
        else:
            return [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i + 1, len(entities))]
