import pandas as pd
from tqdm import tqdm
import json
import argparse
import torch
import transformers
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from openai import OpenAI
from datasets.utils.logging import disable_progress_bar
from utils import (pd_format_date, get_trading_days, map_dates_to_values, Bert_embeder, x_month_ago, argument_keyword, BGE_embeder,
                   get_first_and_last_trading_days, format_symbol)

sys.path.insert(0, sys.path[0] + "/../")
from models.Graph_RAG import graph_rag
from models.temp_walk_RAG import temporal_walk_rag

pd.options.mode.chained_assignment = None
disable_progress_bar()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# max_memory_mapping = {1: "10GB", 2: "10GB", 3: "10GB", 4: "10GB", 5: "10GB"}


def ll3_instance(args):
    model_dir = "/data/user/seraveea/research/hugging_face_cache/Meta-Llama-3-8B-Instruct_hf"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    return pipeline


def deepseek_instance(args):
    with open('scripts/DeepSeek_API.txt', 'r') as file:
        api_key = file.read().strip()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client


def main(args):
    torch.compile(mode='reduce-overhead')  # try to speed up the llama3
    dataset = pd.read_pickle(args.source_path)  # we use dataframe instead of datasets format in graph exp
    dataset['published time'] = dataset['published time'].apply(lambda x: pd_format_date(x))
    dataset['symbol'] = dataset['symbol'].apply(lambda x: x.strip("'"))
    trading_days_cal = get_trading_days(args.first_trading_day, args.last_trading_day, args.language)
    result_df_list = []
    doc_dict_list = []
    if args.backbone == 'llama3':
        llm = ll3_instance(args)
    elif args.backbone == 'gpt4o':
        llm = args.backbone
    else:
        llm = 'no model'
    if args.language == 'zh':
        embeder = BGE_embeder(args.device)
        maper = pd.read_csv('data/index_csi100.csv')
        maper['symbol'] = maper['symbol'].str.strip("'")

    else:
        embeder = Bert_embeder()
        maper = pd.read_csv('data/index_nasdaq.csv')
    ts = pd.read_pickle(args.ts_path)
    summary = pd.read_pickle(args.summary_path)
    rag_agent = temporal_walk_rag(llm, ts, summary, maper, embeder, args.preload_doc_path, style=args.style, language=args.language)
    if args.language == 'zh':
        static_knowledge = pd.read_pickle('data/chn_static_knowledge.pkl')
    else:
        static_knowledge = pd.read_pickle('data/static_knowledge.pkl')
    for trading_day in tqdm(trading_days_cal):
        # don't contain current day's news
        start_day = x_month_ago(trading_day, int(args.lookback))
        sub_dataset = dataset[(dataset['published time'] < trading_day) & (dataset['published time'] > start_day)]
        # here we add static knowledge, default published time is the first day of start_day
        static_knowledge['published time'] = start_day
        date_dict = map_dates_to_values(start_day, trading_day, 64)
        for symbol in tqdm(maper['symbol'].tolist(), leave=False):
            result, retrieved_doc = rag_agent.temporal_walk_reply(symbol, trading_day,
                                                                  sub_dataset, static_knowledge,
                                                                  date_dict, int(args.topk))
            result_df_list.append(result)
            doc_dict_list.append(retrieved_doc)
    rag_agent.log_time(args.runtime_recording_path)
    rag_agent.log_infer_time(args.runtime_recording_path)
    if args.backbone == 'no model':
        print('no LLM specified, only store the retrieval results.')
    else:
        result_df = pd.concat(result_df_list, axis=0)
        result_df.to_pickle(args.result_path)
    if not args.preload_doc_path:
        retrieve_df = pd.DataFrame(doc_dict_list)
        retrieve_df.to_pickle(args.doc_path)

    # ------recording all arguments into json
    args_dict = {'parsed_args': vars(args)}
    with open(args.args_path, 'a') as f:
        json.dump(args_dict, f)
        f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_trading_day', default='2024-01-01')
    parser.add_argument('--last_trading_day', default='2024-07-01')
    parser.add_argument('--lookback', default=1)
    parser.add_argument('--candidated', default=20)
    parser.add_argument('--backbone', default='llama3',
                        help='several mode, llama3/GPT/FinPTForecaster/deepseek/no model')
    parser.add_argument('--result_path', default='output/nograph_summary_GCS_Q1.pkl',
                        help='the path of saving llm reply result')
    parser.add_argument('--doc_path', default='output/retrieval_only/nograph_summary_GCS_doc_Q1.pkl',
                        help='the path of saving retrieved file list')
    parser.add_argument('--source_path', default='data/new_llama3/nasdaq_summary24.pkl',
                        help='the path of doc pools')
    parser.add_argument('--ts_path', default='data/nasdaq_ts_2024.pkl')
    parser.add_argument('--summary_path', default='data/ndx100_business_summary.pkl',
                        help='the path of static knowledge')
    parser.add_argument('--language', default='zh', choices=['zh', 'en'])
    parser.add_argument('--runtime_recording_path', default='logs/gcs_run_time.log')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--topk', default=10)
    parser.add_argument('--style', default='direct', help='direct, indirect3')
    parser.add_argument('--preload_doc_path', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    myargs = parse_args()
    kw = argument_keyword(myargs.result_path)
    myargs.doc_path = 'output/retrieval_result/' + kw + '_doc.pkl'
    myargs.runtime_recording_path = 'logs/' + kw + '.log'
    myargs.args_path = 'logs/' + kw + '.json'

    main(myargs)
