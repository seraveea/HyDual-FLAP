"""
after generating one sentence summary of every document
using hugging face pretrained bert to embedding the summary
to save array, use pickle format to replace csv
"""
from utils import Bert_embeder
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
tqdm.pandas()


def main(myargs):
    target_df = pd.read_pickle(myargs.doc_path)
    embeder = Bert_embeder(myargs.device)
    target_df['embedding'] = target_df['one_sentence'].progress_apply(lambda x: np.array(embeder.embed(x).cpu()))
    target_df.to_pickle(myargs.result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', default='data/nasdaq_summary24.pkl')
    parser.add_argument('--result_path', default='data/nasdaq_summary24.pkl')
    parser.add_argument('--device', default='cuda:0')

    myargs = parser.parse_args()
    return myargs


if __name__ == '__main__':
    args = parse_args()
    main(args)
