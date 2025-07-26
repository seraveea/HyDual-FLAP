"""
after generating one sentence summary of every document
using hugging face pretrained bert to embedding the summary
to save array, use pickle format to replace csv
"""
from utils import Bert_embeder, BGE_embeder
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

tqdm.pandas()

os.environ["HF_HOME"] = "/data/user/seraveea/research/hugging_face_cache"
huggingface_cache_path = "/data/user/seraveea/research/hugging_face_cache"
os.environ['HF_HUB_OFFLINE'] = '1'

def main(myargs):
    target_df = pd.read_pickle(myargs.doc_path)
    if args.language == 'zh':
        embeder = BGE_embeder(myargs.device)
    else:
        embeder = Bert_embeder(myargs.device)
    target_df['embedding'] = target_df['one_sentence'].progress_apply(lambda x: embeder.embed(x))
    target_df.to_pickle(myargs.result_path)


def build_prompt(line):
    return f"""Title: {line['Title']}\nDescription: {line['Description']}\nLong Description: {line['Long description']}
    """


def main_raw(myargs):
    target_df = pd.read_pickle(myargs.doc_path)
    embeder = Bert_embeder()
    target_df['embedding'] = target_df.progress_apply(lambda x: np.array(embeder.embed(build_prompt(x))), axis=1)
    target_df.to_pickle(myargs.result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', default='data/news_llama3/csi100_summary.pkl')
    parser.add_argument('--result_path', default='data/news_llama3/csi100_summary.pkl')
    parser.add_argument('--language', default='zh')
    parser.add_argument('--device', default='cuda:0')

    myargs = parser.parse_args()
    return myargs


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # main_raw(args)
