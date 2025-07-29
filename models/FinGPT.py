"""
adopt from FinGPT repo: https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster
"""
from datasets import load_dataset
from transformers import AutoTokenizer
import uuid
import re
import torch
import pandas as pd
from models.noGraph_RAG import vanilla_RAG
from scripts.utils import *


class FinGPT_forecaster(vanilla_RAG):
    def __init__(self, client, time_series, summary, maper, embeder, style, language):
        super().__init__(client, time_series, summary, maper, embeder, preload_doc_path=False, style=style, language=language)
        self.tokenizer = AutoTokenizer.from_pretrained('[your llama2 model path]')

    def rag_reply(self, symbol, date, sub_dataset, topk, **kwargs):
        query = self.default_query(symbol, date)
        q_vector = np.array(self.embeder.embed(query))
        scores, retrieved_doc = sub_dataset.get_nearest_examples('embedding', q_vector, k=topk)
        separate_dict = separate_dictionary(retrieved_doc, topk, scores)
        time_series, ground_truth = self.find_ground_truth(symbol, date)
        user_prompt = self.generate_prompt_fingpt(separate_dict, symbol, date, time_series)
        if self.language == 'zh':
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        else:
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        return self.client_reply(symbol, date, ground_truth, separate_dict, prompt)

    def gcs_reply(self, symbol, date, sub_dataset, topk, **kwargs):
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                docs = sub_dataset.filter(lambda e: self.retrieve_based_on_given_list(e, doc_list))
                separate_dict = separate_dictionary(docs.to_dict(), topk)
            except:
                print('no preload files error')
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                     'N/A',
                                     [{'url': 'empty'}],
                                     0)
        else:
            doc_dataset = sub_dataset.filter(lambda x: (x['symbol'] == symbol))
            sort_doc = doc_dataset.sort('published time')
            if len(sort_doc) < topk:
                print(f'not enough documents for {symbol} on {date}, only {len(sort_doc)} documents')
                remains = sub_dataset.filter(lambda x: (x['symbol'] != symbol))
                sort_remains = remains.sort('published time')
                remain_dict = separate_dictionary(sort_remains[:topk - len(sort_doc)], topk - len(sort_doc))
                separate_dict = separate_dictionary(sort_doc[:], len(sort_doc)) + remain_dict
            else:
                separate_dict = separate_dictionary(sort_doc[-topk:], topk)
        time_series, ground_truth = self.find_ground_truth(symbol, date)
        user_prompt = self.generate_prompt_fingpt(separate_dict, symbol, date, time_series)
        if self.language == 'zh':
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        else:
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        return self.client_reply(symbol, date, ground_truth, separate_dict, prompt)

    def random_reply(self, symbol, date, sub_dataset, topk):
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                docs = sub_dataset.filter(lambda e: self.retrieve_based_on_given_list(e, doc_list))
                separate_dict = separate_dictionary(docs.to_dict(), topk)
            except:
                print('no preload files error')
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                     'N/A',
                                     [{'url': 'empty'}],
                                     0)
        else:
            sort_doc = sub_dataset.sort('published time')
            if len(sort_doc) < topk:
                separate_dict = separate_dictionary(sort_doc[:], len(sort_doc))
            else:
                indices = random.sample(range(len(sort_doc)), topk)
                separate_dict = [sort_doc[i] for i in indices]

        time_series, ground_truth = self.find_ground_truth(symbol, date)
        user_prompt = self.generate_prompt_fingpt(separate_dict, symbol, date, time_series)
        if self.language == 'zh':
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        else:
            prompt = self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + user_prompt + self.E_INST
        return self.client_reply(symbol, date, ground_truth, separate_dict, prompt)

    def client_reply(self, symbol, date, ground_truth, separate_dict, prompt):
        """
        client reply for FinGPT forecaster
        """
        prompt_len = len(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=False, max_length=4096, truncation=True)
        inputs = {key: value.to(self.client.device) for key, value in inputs.items()}
        res = self.client.generate(
            **inputs, max_length=5120, do_sample=True, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)

        output = self.tokenizer.decode(res[0], skip_special_tokens=True)
        return self.get_dict(symbol, date, output[prompt_len:], ground_truth, separate_dict, prompt_len)

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    # SYSTEM_PROMPT = """You are a seasoned stock market analyst.
    # Your task is to list the positive developments and potential concerns
    # for companies based on relevant news and basic financials from the past weeks,
    # then provide an analysis and prediction for the companies' stock price movement for the upcoming week.
    # Your answer format should be as follows:\n
    # [Positive Developments]:\n1. ...\n
    # [Potential Concerns]:\n1. ...\n
    # [Prediction & Analysis]:\n...\n"
    # """
    SYSTEM_PROMPT = """
    You will given a series of news with released date and content.
    Please evaluate given stock's price change ONLY based on the provided news from given perspectives:
    1. market share: the percentage of total sales a company captures within its industry.
    2. company strategies: long-term plans that guide business in future development.
    3. products performance: the quality and profitability of company product or services.
    4. industry status: the overall marco situation of the company's sector or industry.
    5. investor sentiment: the overall attitude or mood of investors toward the company.
    6. stock risk: potential for financial loss due to market volatility, economic changes, or company performance.
    7. competitor status: the current position, strategies, strengths, and weaknesses of rival companies within the industry.
    8. supplier status: performance, reliability, capacity, and financial health of the company’s suppliers, impacting the supply chain.
    9. innovation sustainability: the ability to maintain and develop new, impactful innovations over time.
    You need to forecast next trading day stock return for stock on date of user given. 
    The next day stock return is represented by bins "D5+", "D5", "D4", "D3", "D2", "D1", "U1", "U2", "U3", "U4", "U5", "U5+", 
    where "D5+" means price dropping more than 5%, D5 means price dropping between 4% and 5%, 
    "D4" means price dropping between 3% and 4%, "U5+" means price rising more than 5%, 
    "U5" means price rising between 4% and 5%, "D4" means price rising between 3% and 4%, etc..
    """

    chn_SYSTEM_PROMPT = """
    您将收到一系列包含发布日期和内容的新闻。
    请仅根据提供的新闻从以下角度评估股票价格变化：
    1. 市场份额
    2. 公司战略
    3. 产品表现
    4. 行业状况
    5. 投资者情绪
    6. 股票风险
    7. 竞争对手状况
    8. 供应商状况
    9. 创新可持续性
    您需要预测用户指定日期股票下周的收益。
    下周股票收益用“D5+”、“D5”、“D4”、“D3”、“D2”、“D1”、“U1”、“U2”、“U3”、“U4”、“U5”、“U5+”等符号表示，
    其中“D5+”表示价格下跌超过5%，D5表示价格下跌在4%到5%之间，
    “D4”表示价格下跌在3%到4%之间，“U5+”表示价格上涨超过5%，
    “U5”表示价格上涨在4%到5%之间，“D4”表示价格上涨在3%到4%之间，等等。
    根据您的观察，在第一行中，将下一个交易日的股票收益预测放置在特殊符号$$到$$之间。
    """    

    def generate_prompt_fingpt(self, doc_list, symbol, date, ts):
        symbol_summary = self.business_summary.loc[(0, symbol)]['Summary']
        if len(doc_list) == 0:
            return f'STOCK: {symbol}; DATE: {date}'
        else:
            sentences_list = [f"STOCK: {symbol}; DATE: {date};\n NEWS:"]
            if 'Title' in doc_list[0].keys():
                for docu in doc_list:
                    temp = f"""<date>{docu['published time']}</date><doc>Title: {docu['Title']}\n
                    Description: {docu['Description']}\nLong Description: {docu['Long description']}</doc>
                    """
                    sentences_list.append(temp)
            else:
                sentences_list.append(f"<Company Summary>{symbol_summary}</Company Summary>\n")
                for docu in doc_list:
                    temp = f"[News Date]: {docu['published time']}\n[Summary]:{deal_with_nan(docu['one_sentence'])}\n"
                    sentences_list.append(temp)

            # sentences_list.append(f'<TS>{ts}</TS>')
            return ' '.join(sentences_list)
