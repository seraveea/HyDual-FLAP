import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import argparse
import torch
import transformers
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.noGraph_RAG import vanilla_RAG
from models.FinGPT import FinGPT_forecaster
from datasets.utils.logging import disable_progress_bar
from utils import format_date, get_trading_days, Bert_embeder, x_month_ago, argument_keyword, get_first_and_last_trading_days

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
disable_progress_bar()
# max_memory_mapping = {3: "10GB", 4: "10GB", 5: "10GB"}


def ll3_instance(args):
    model_dir = "/export/data/RA_Work/seraveea/llama3/Meta-Llama-3-8B-Instruct_hf"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
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
        '/export/data/RA_Work/seraveea/Llama-2-7b-chat-hf',
        trust_remote_code=True,
        device_map=args.device,
        # max_memory=max_memory_mapping,
        # device_map="auto",
        torch_dtype=torch.float16,  # optional if you have enough VRAM
    )
    return PeftModel.from_pretrained(base_model, 'fingpt-forecaster_dow30_llama2-7b_lora')


def main(args):
    torch.compile(mode='reduce-overhead')  # try to speed up the llama3
    dataset = Dataset.from_pandas(pd.read_pickle(args.source_path), split='train')
    dataset = dataset.map(format_date)
    trading_days_cal = get_trading_days(args.first_trading_day, args.last_trading_day)
    result_df_list = []
    doc_dict_list = []
    if args.backbone == 'llama3':
        llm = ll3_instance(args)
    elif args.backbone == 'fingpt':
        llm = fingpt_instance(args).eval()
    elif args.backbone == 'gpt4o':
        llm = args.backbone
    elif args.backbone == 'deepseek':
        llm = ds_instance(args)
    else:
        # if no backbone specified
        llm = 'no model'
    bert_embeder = Bert_embeder()
    maper = pd.read_csv('data/nasdaq_dict.csv')
    ts = pd.read_pickle(args.ts_path)
    summary = pd.read_pickle(args.summary_path)
    if args.backbone == 'fingpt':
        rag_agent = FinGPT_forecaster(llm, ts, summary, maper, bert_embeder, style=args.style)
    else:
        rag_agent = vanilla_RAG(llm, ts, summary, maper, bert_embeder, args.preload_doc_path, style=args.style)

    for trading_day in tqdm(trading_days_cal):
        # don't contain current day's news
        start_day = x_month_ago(trading_day, int(args.lookback))
        sub_dataset = dataset.filter(lambda x: (x['published time'] < trading_day) & (x['published time'] > start_day))
        if args.model_name == 'RAG' and not args.preload_doc_path:
            sub_dataset = sub_dataset.add_faiss_index(column='embedding')
        for symbol in tqdm(maper['symbol'].tolist(), leave=False):
            if args.model_name == 'RAG':
                result, retrieved_doc = rag_agent.rag_reply(symbol, trading_day, sub_dataset,
                                                            int(args.topk), args.backbone)
            elif args.model_name == 'GCS':
                result, retrieved_doc = rag_agent.gcs_reply(symbol, trading_day, sub_dataset,
                                                            int(args.topk), args.backbone)
            elif args.model_name == 'TempRALM':
                result, retrieved_doc = rag_agent.gcs_reply(symbol, trading_day, sub_dataset,
                                                            int(args.topk), args.backbone)

            else:
                result, retrieved_doc = rag_agent.random_reply(symbol, trading_day, sub_dataset, int(args.topk),
                                                               args.backbone)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_trading_day', default='2024-01-01')
    parser.add_argument('--last_trading_day', default='2024-07-01')
    parser.add_argument('--model_name', default='RAG')
    parser.add_argument('--lookback', default=1)
    parser.add_argument('--backbone', default='llama3',
                        help='several mode, llama3/GPT/FinPTForecaster/deepseek/no model')
    parser.add_argument('--result_path', default='output/DeepSeek/nograph_summary_rag_Q1.pkl',
                        help='the path of saving llm reply result')
    parser.add_argument('--doc_path', default='output/retrieval_only/nograph_summary_rag_doc_Q1.pkl',
                        help='the path of saving retrieved file list')
    parser.add_argument('--source_path', default='data/new_llama3/nasdaq_summary24.pkl',
                        help='the path of doc pools')
    parser.add_argument('--ts_path', default='data/nasdaq_ts_2024.pkl')
    parser.add_argument('--summary_path', default='data/ndx100_business_summary.pkl',
                        help='the path of static knowledge')
    parser.add_argument('--runtime_recording_path', default='logs/gcs_run_time.log')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--topk', default=10)
    parser.add_argument('--style', default='indirect3')
    parser.add_argument('--preload_doc_path', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    myargs = parse_args()
    kw = argument_keyword(myargs.result_path)
    myargs.doc_path = 'output/retrieval_result/'+kw+'_doc.pkl'
    myargs.runtime_recording_path = 'logs/' + kw + '.log'
    myargs.args_path = 'logs/' + kw + '.json'
    main(myargs)
