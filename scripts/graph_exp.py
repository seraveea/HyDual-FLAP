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
from utils import (pd_format_date, get_trading_days, map_dates_to_values, Bert_embeder, x_month_ago, argument_keyword,
                   get_first_and_last_trading_days)

sys.path.insert(0, sys.path[0] + "/../")
from models.Graph_RAG import graph_rag
from models.temp_walk_RAG import temporal_walk_rag
from models.FinGPT import FinGPT_forecaster

pd.options.mode.chained_assignment = None
disable_progress_bar()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# max_memory_mapping = {1: "10GB", 2: "10GB", 3: "10GB", 4: "10GB", 5: "10GB"}


def ll3_instance(args):
    model_dir = "[llama3 model path]"
    pipeline = transformers.pipeline("text-generation", model=model_dir,
                                     torch_dtype=torch.float16, device_map=args.device)
    return pipeline


def ds_instance(args):
    # Load model directly
    pipeline = transformers.pipeline("text-generation",
                                     model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                     torch_dtype=torch.float16,
                                     device_map=args.device)
    return pipeline


def fingpt_instance(args):
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        '[llama2 model path]',
        trust_remote_code=True,
        # max_memory=max_memory_mapping,
        device_map=args.device,
        torch_dtype=torch.float16,  # optional if you have enough VRAM
    )
    return PeftModel.from_pretrained(base_model, 'fingpt-forecaster_dow30_llama2-7b_lora')


def main(args):
    torch.compile(mode='reduce-overhead')  # try to speed up the llama3
    dataset = pd.read_pickle(args.source_path)  # we use dataframe instead of datasets format in graph exp
    dataset['published time'] = dataset['published time'].apply(lambda x: pd_format_date(x))
    trading_days_cal = get_trading_days(args.first_trading_day, args.last_trading_day)
    result_df_list = []
    doc_dict_list = []
    if args.backbone == 'llama3':
        llm = ll3_instance(args)
    elif args.backbone == 'fingpt':
        llm = fingpt_instance(args).eval()
    elif args.backbone == 'gpt4o':
        llm = args.backbone
    else:
        llm = 'no model'
    bert_embeder = Bert_embeder()
    maper = pd.read_csv('data/nasdaq_dict.csv')
    ts = pd.read_pickle(args.ts_path)
    summary = pd.read_pickle(args.summary_path)
    if args.model_name == 'temp_walk':
        rag_agent = temporal_walk_rag(llm, ts, summary, maper, bert_embeder, args.preload_doc_path, style=args.style)
    else:  # not running sampling methods
        if args.backbone == 'fingpt':
            rag_agent = FinGPT_forecaster(llm, ts, summary, maper, bert_embeder, style=args.style)
        else:
            rag_agent = graph_rag(llm, ts, summary, maper, bert_embeder, args.preload_doc_path, style=args.style)

    static_knowledge = pd.read_pickle('data/static_knowledge.pkl')
    for trading_day in tqdm(trading_days_cal):
        # don't contain current day's news
        start_day = x_month_ago(trading_day, int(args.lookback))
        sub_dataset = dataset[(dataset['published time'] < trading_day) & (dataset['published time'] > start_day)]
        # here we add static knowledge, default published time is the first day of start_day
        static_knowledge['published time'] = start_day
        date_dict = map_dates_to_values(start_day, trading_day, 64)
        for symbol in tqdm(maper['symbol'].tolist(), leave=False):
            if args.model_name == 'RAG':
                result, retrieved_doc = rag_agent.graph_rag_reply(symbol, trading_day, sub_dataset, static_knowledge,
                                                                  int(args.topk), args.backbone)
            elif args.model_name == 'temp_walk':
                result, retrieved_doc = rag_agent.temporal_walk_reply(symbol, trading_day,
                                                                      sub_dataset, static_knowledge,
                                                                      date_dict, int(args.topk), int(args.lookback))
            else:
                result, retrieved_doc = None, None
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
    parser.add_argument('--model_name', default='RAG',
                        help="choose one from RAG, GCS, temp_walk, random_walk")
    parser.add_argument('--backbone', default='llama3')
    parser.add_argument('--lookback', default=1)
    parser.add_argument('--result_path', default='output/graph_baseline/graph_RAG_Q1.pkl')
    parser.add_argument('--doc_path', default='output/retrieval_only/graph_RAG_doc_Q1.pkl')
    parser.add_argument('--source_path', default='data/new_llama3/nasdaq_summary24.pkl')
    parser.add_argument('--summary_path', default='data/ndx100_business_summary.pkl')
    parser.add_argument('--runtime_recording_path', default='logs/graph_rw_run_time.log')
    parser.add_argument('--args_path', default='logs/graph_rw_DS.json')
    parser.add_argument('--ts_path', default='data/nasdaq_ts_2024.pkl')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--topk', default=10)
    parser.add_argument('--style', default='direct',
                        help="choose one from direct, indrect3")
    parser.add_argument('--candidated', default=20)
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
