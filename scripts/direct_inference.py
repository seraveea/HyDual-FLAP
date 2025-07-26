import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef

def convert_to_category(x, counter=None, gpt_flag=False):
    if counter is None:
        counter = {'normal':0, 'only_rise':0, 'only_fall':0, 'invalid':0, 'global_valid':0}
    lines = x.split("\n")
    if gpt_flag:
        last_line = lines[0].strip()
    else:
        last_line = lines[-1].strip()
    categories = ["U1", "U2", "U3", "U4", "U5", "U5+","D1", "D2", "D3","D4", "D5","D5+" ]
    for category in categories:
        if category in last_line:
            counter['normal'] += 1
            return category
    if "rise" in last_line or "increase" in last_line:
        if any(char.isdigit() for char in last_line):
            num = int(''.join(filter(str.isdigit, last_line)))
            if num < 5:
                counter['normal'] += 1
                return f"U{num + 1}"
            else:
                counter['normal'] += 1
                return "U5+"
        else:
            counter['only_rise'] += 1
            return "U1"
    elif "fall" in last_line or "decrease" in last_line:
        if any(char.isdigit() for char in last_line):
            num = int(''.join(filter(str.isdigit, last_line)))
            if num < 5:
                counter['normal'] += 1
                return f"D{num + 1}"
            else:
                counter['normal'] += 1
                return "D5+"
        else:
            counter['only_rise'] += 1
            return "D1"
    else:
        # the target line does not contain any of the categories
        # search for the whole text
        # if gpt_flag:
        #     for category in categories:
        #       if category in x:
        #           counter['global_valid'] += 1
        #           return category
        counter['invalid'] += 1
        return "D1"

def assign_ground_truth(exp_result, ts_2024):
    def assign_t(time_series, x):
        date = x['date']
        symbol = x['symbol']
        temp = time_series[(time_series['Date'] == date) & (time_series['Symbol'] == symbol)]
        if temp.shape[0] != 1:
            return 'N/A'
        else:
            return temp['daily_return_bins'].tolist()[0]
    exp_result['ground_truth'] = exp_result.apply(lambda row: assign_t(ts_2024, row), axis=1)
    return exp_result

def cat2num(x):
    dic = ['D5+', 'D5', 'D4', 'D3', 'D2', 'D1', 'U1', 'U2', 'U3', 'U4', 'U5', 'U5+']
    for i in range(len(dic)):
        if x == dic[i]:
            return i
    return 5


def preprocess_exp_result(exp_result, ts_2024, flag='CHN', gpt_flag=False):
    counter = {'normal':0, 'only_rise':0, 'only_fall':0, 'invalid':0, 'global_valid':0}
    if gpt_flag:
        exp_result['pred'] = exp_result['result'].apply(lambda x: convert_to_category(x, counter, True))
    else:
        exp_result['pred'] = exp_result['result'].apply(lambda x: convert_to_category(x, counter))
    if flag == 'CHN':
        ts_2024['Date'] = ts_2024['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    exp_result = assign_ground_truth(exp_result, ts_2024)
    exp_result = exp_result[exp_result['ground_truth'] != 'N/A']
    exp_result['mc_pred'] = exp_result['pred'].apply(cat2num)
    exp_result['mc_gt'] = exp_result['ground_truth'].apply(cat2num)
    preds_file = exp_result[['symbol','date','mc_pred','mc_gt']]
    preds_file.to_csv(preds_name, index=False)
    return exp_result, counter

def evaluate_exp_result(exp_result):
    gt_list = exp_result['mc_gt'].values.tolist()
    pred = exp_result['mc_pred'].values.tolist()
    multiclass_metrics = {
        "accuracy": accuracy_score(gt_list, pred),
        "f1": f1_score(gt_list, pred, average='weighted'),
    }
    return multiclass_metrics

if __name__ == "__main__":
    # 11*2
    file_list = [
        "output/chn/Chatqa_chn.parquet",
        "output/chn/DeepSeek_chn.parquet",
        "output/chn/FinGPT_chn.parquet",
        "output/chn/GPT4o_chn.parquet",
        "output/chn/HyDual_only_chn.parquet",
        "output/chn/Llama3_chn.parquet",
        "output/chn/RAG_DeepSeek_chn.parquet",
        "output/chn/RAG_FinGPT_chn.parquet",
        "output/chn/RAG_GPT4o_chn.parquet",
        "output/chn/RAG_Llama3_chn.parquet",
        "output/chn/tempralm_chn.parquet",
        "output/eng/Chatqa.parquet",
        "output/eng/DeepSeek.parquet",
        "output/eng/FinGPT.parquet",
        "output/eng/GPT4o.parquet",
        "output/eng/HyDual_only.parquet",
        "output/eng/Llama3.parquet",
        "output/eng/RAG_DeepSeek.parquet",
        "output/eng/RAG_FinGPT.parquet",
        "output/eng/RAG_GPT4o.parquet",
        "output/eng/RAG_Llama3.parquet",
        "output/eng/tempralm.parquet",
    ]
    result_list = []
    for file_name in tqdm(file_list):
        if file_name.endswith('.pkl'):
            exp_result = pd.read_pickle(file_name, compression='infer')
        else:
            exp_result = pd.read_parquet(file_name)
        print(f"Processing file: {file_name}")
        
        if 'chn' in file_name:
            flag = 'CHN'
        else:
            flag = 'USA'

        if flag == 'CHN':
            ts_2024 = pd.read_pickle('data/csi100_ts_2024.pkl')
        else:
            ts_2024 = pd.read_pickle('data/nasdaq_ts_2024.pkl')

        if file_name.endswith('.pkl'):
            if flag == 'CHN':
                preds_name = 'preds/chn/' + file_name.split('/')[-1].replace('.pkl', '_preds.csv')
            else:
                preds_name = 'preds/eng/' + file_name.split('/')[-1].replace('.pkl', '_preds.csv')
        else:
            if flag == 'CHN':
                preds_name = 'preds/chn/' + file_name.split('/')[-1].replace('.parquet', '_preds.csv')
            else:
                preds_name = 'preds/eng/' + file_name.split('/')[-1].replace('.parquet', '_preds.csv')
        if 'GPT' in file_name and 'chn' in file_name:
            gpt_flag = True
        else:
            gpt_flag = False
        exp_result, counter = preprocess_exp_result(exp_result, ts_2024, flag=flag, gpt_flag=gpt_flag)
        multiclass_metrics = evaluate_exp_result(exp_result)
        print(multiclass_metrics)
        print(counter)
        multiclass_metrics['index']= file_name.split('/')[-1].replace('.parquet', '')
        result_list.append(multiclass_metrics)
    result_df = pd.DataFrame(result_list)
    print(result_df)