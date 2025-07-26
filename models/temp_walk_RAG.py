import pandas as pd
import random
from collections import Counter
from .Graph_RAG import graph_rag
import uuid
import time
from scripts.utils import x_month_ago, map_dates_to_valuesv2
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, sys.path[0] + "/../")
from scripts.utils import generate_prompt_ll3


class temporal_walk_rag(graph_rag):
    def __init__(self, client, time_series, summary, maper, embeder, preload_doc_path, style, language):
        super().__init__(client, time_series, summary, maper, embeder, preload_doc_path, style, language)
        self.graph = None
        self.stat_dict = None
        self.num_rw = 100  # the total number of random walk
        self.depth_rw = 10  # the depth of one random walk

    def temporal_walk_reply(self, symbol, date, sub_dataset, static_data, time_dict, topk):
        # **********
        start_time = time.time()
        # **********
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                separate_dict = sub_dataset[sub_dataset['url'].isin(doc_list)].to_dict(orient='records')
            except:
                print('error')
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                     'N/A',
                                     [{'url': 'empty'}],
                                     0)
        else:
            sub_dataset['uuid'] = sub_dataset['published time'].apply(lambda x: uuid.uuid4())
            static_data['uuid'] = static_data['published time'].apply(lambda x: uuid.uuid4())
            separate_dict = self.random_walk(symbol, sub_dataset, static_data, time_dict, date, topk)
        time_series, ground_truth = self.find_ground_truth(symbol, date)
        retrieval_prompt = generate_prompt_ll3(separate_dict, date, symbol, time_series, self.language)
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

    def random_walk(self, symbol, data, static_data, time_dict, date, topk):
        """
        :param topk: how many documents retrieved
        :param reserve_for_event: how many documents for node itself
        :param time_dict: time vector dictionary
        :param date: target day
        :param symbol: center node
        :param data: a dataframe with recent news
        :param static_data: static data
        :return:
        """
        center = self.maper[symbol]

        data = data[data['refined_entities_len'] < 6]
        query = self.default_query(symbol, date)
        center_embedding = np.array(self.embeder.embed(query))
        # ----------------------temporal random walk----------------------------
        # half-half
        start_day = x_month_ago(date, 1)
        time_dict = map_dates_to_valuesv2(start_day, date, 64)
        data['l2_distance'] = data['embedding'].apply(lambda x: np.linalg.norm(x - center_embedding))
        data['l2_distance'] = self.normalize(data['l2_distance'])  # 0~1
        data['score_weight'] = 1.001 - data['l2_distance']  # 0~1
        data['time_weight'] = data['published time'].apply(lambda x: time_dict[x])  # 0~1
        data['weight'] = 0.5*data['time_weight']+0.5*data['score_weight']  # 0~2

        static_data['l2_distance'] = static_data['embedding'].apply(lambda x: np.linalg.norm(x - center_embedding))
        static_data['l2_distance'] = self.normalize(static_data['l2_distance'])
        static_data['score_weight'] = 1.001 - static_data['l2_distance']
        static_data['time_weight'] = data['time_weight'].mean()
        static_data['weight'] = 0.5*static_data['time_weight']+0.5*static_data['score_weight']
        # -------------------------7/3----------------------------------------------
        # data['weight'] = 0.7*data['time_weight']+0.3*data['score_weight']  # 0~2
        # static_data['weight'] = 0.7*static_data['time_weight']+0.3*static_data['score_weight']
        # -------------------------3/7----------------------------------------------
        # data['weight'] = 0.3*data['time_weight']+0.7*data['score_weight']  # 0~2
        # static_data['weight'] = 0.3*static_data['time_weight']+0.7*static_data['score_weight']

        tkg = nx.MultiGraph()
        for index, row in static_data.iterrows():
            tkg.add_edge(row['Subject'], row['Object'], label=row['uuid'], weight=row['weight'])
        for index, row in data.iterrows():
            pairs = self.create_pairs(row['refined_entities'])
            for pair in pairs:
                tkg.add_edge(pair[0], pair[1], label=row['uuid'], weight=row['weight'])
        data = pd.concat([data, static_data])  # concat static_data into data

        edge_path_container = {}  # record appeared paths
        edge_list = {}  # record appeared edges
        # repeat num_rw times
        for i in range(self.num_rw):
            path, edge_path = self.weighted_random_walk(tkg, center, self.depth_rw)
            # the path id is generated by a join of each edge id
            edge_path_id = '$$'.join([str(x['label']) for x in edge_path])
            for edge in edge_path:
                if edge['label'] in edge_list.keys():
                    edge_list[edge['label']] += 1
                else:
                    edge_list[edge['label']] = 1
            if edge_path_id in edge_path_container.keys():
                edge_path_container[edge_path_id] += 1
            else:
                edge_path_container[edge_path_id] = 1

        # a sorted dictionary for all edges
        sorted_dictionary = dict(sorted(edge_list.items(), key=lambda item: item[1], reverse=True))
        # sorted_dictionary keys are UUID
        sorted_path_dict = dict(sorted(edge_path_container.items(), key=lambda item: item[1], reverse=True))
        # sorted_path_dict keys are multiple UUID string
        separate_dict = self.choose_rw_edges(sorted_dictionary, sorted_path_dict, data, tkg, topk)
        return separate_dict

    def weighted_random_walk(self, graph, start_node, num_steps, restart_threshold=0.2):
        """Perform a weighted random walk on the graph."""
        path = [start_node]  # include start node first
        edge_chosen = []
        current_node = start_node  # current node
        previous_node = None

        for _ in range(num_steps):
            if random.random() < restart_threshold:
                # 20% percentage to break and return current path
                break
            else:
                neighbors = list(graph.neighbors(current_node))  
                if previous_node is not None:
                    neighbors.remove(previous_node)  
                if not neighbors:
                    break  # No more neighbors to visit
                edges = [list(graph.get_edge_data(neighbor, current_node).values()) for neighbor in neighbors]
                # all 1-hop edges
                next_index, edge = self.weighted_random_choice(edges)  # find next node by random walk
                previous_node = current_node  # assign previous node
                current_node = neighbors[next_index]  # change current node
                path.append(current_node)
                edge_chosen.append(edge)

        return path, edge_chosen

    @staticmethod
    def weighted_random_choice(weights):
        """
        :param weights: a list of dictionary
        :return: the index and edge weights
        """
        total = 0
        for edges in weights:
            for edge in edges:
                total += edge['weight']
        # total = sum([w['weight'] for w in weights])
        r = random.uniform(0, total)
        upto = 0
        for i, edges in enumerate(weights):
            for edge in edges:
                if upto + edge['weight'] >= r:
                    return i, edge
                upto += edge['weight']
        print(weights)
        assert False, "Shouldn't get here"

    @staticmethod
    def normalize(values):
        min_val = np.min(values)
        max_val = np.max(values)
        return [(value - min_val) / (max_val - min_val) for value in values]

    def choose_rw_edges(self, sorted_dictionary, sorted_path_dict, data, tkg, topk):
        if len(sorted_dictionary) < topk:
            # extreme case, the whole graph is smaller than required num of docs, then directly return all edges
            sub_data = data.merge(pd.DataFrame(list(sorted_dictionary.items()), columns=['uuid', 'in_rw']), on=['uuid'])
            self.graph = tkg
            sub_data = sub_data.sort_values(by=['in_rw', 'published time'], ascending=False).drop_duplicates(['url'])
            separate_dict = sub_data.to_dict(orient='records')
        else:
            retrieved_files = set()
            edge_index = 0
            while len(retrieved_files) < topk and edge_index < len(list(sorted_dictionary.keys())):
                # still have un checked edge and container not full
                target_edge = str(list(sorted_dictionary.keys())[edge_index])

                def helper(r_list, e):
                    for temp_i, string in enumerate(r_list):
                        if e == string:
                            return r_list[:temp_i + 1]
                    return None

                for path in sorted_path_dict.keys():
                    recover_list = path.split("$$")
                    temp_path = helper(recover_list, target_edge)
                    if temp_path is not None:
                        retrieved_files = retrieved_files | set(temp_path)
                        break
                # 下一个edge
                edge_index += 1
            temp = pd.DataFrame([uuid.UUID(s) for s in list(retrieved_files)], columns=['uuid'])
            sub_data = data.merge(temp, on=['uuid'])
            separate_dict = sub_data.to_dict(orient='records')  # use closet documents

        return separate_dict

    def choose_rw_edges_v2(self, sorted_dictionary, sorted_path_dict, data, tkg, topk):
        if len(sorted_dictionary) < topk:
            sub_data = data.merge(pd.DataFrame(list(sorted_dictionary.items()), columns=['uuid', 'in_rw']), on=['uuid'])
            self.graph = tkg
            sub_data = sub_data.sort_values(by=['in_rw', 'published time'], ascending=False).drop_duplicates(['url'])
            separate_dict = sub_data.to_dict(orient='records')
        else:
            retrieved_files = list(sorted_dictionary.keys())[:topk]
            temp = pd.DataFrame(retrieved_files, columns=['uuid'])
            sub_data = data.merge(temp, on=['uuid'])
            separate_dict = sub_data.to_dict(orient='records')  # use closet documents
        return separate_dict
