import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
import numpy as np
import re

def remove_word_and_filter_letters(text, phrase, end_phrase):
    raw_text = text
    lines = text.split('\n')
    container = []
    flag = 0
    for i in range(0, len(lines)):
        if phrase.lower() in lines[i].lower():
            flag = 1
        if end_phrase.lower() in lines[i].lower():
            flag = 0
        if flag:
            container.append(lines[i].lower())
    if len(container) == 0:
        # no words found
        text = 'None'
    else:
        text = ''.join(container)
    text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', text, flags=re.IGNORECASE)
    result = []
    for char in text:
        if char.isalpha():
            result.append(char)
    reply = ''.join(result)
    return reply

def generating_digit(x):
    if x in ['neutral', 'neutral_default', 'mixed', 'moderate', 'notmentioned','notapplicable']:
        return 2
    elif x in ['positive', 'low']:
        return 3
    elif x in ['negative', 'high']:
        return 1

def category(x, phrase, end_phrase):
    temp = remove_word_and_filter_letters(x, phrase, end_phrase)
    for item in ['positive', 'negative', 'neutral', 'mixed', 'high', 'low', 'notmentioned','notapplicable']:
        if item in temp:
            return item
    return 'neutral_default'

def generating_onehot(x):
    if x in ['positive', 'low']:
        return [1, 0, 0]
    elif x in ['negative', 'high']:
        return [0, 1, 0]
    elif x in ['neutral', 'neutral_default', 'mixed', 'moderate', 'notmentioned', 'notapplicable']:
        return [0, 0, 1]
    else:
        return [0, 0, 1]


def helper(x):
    dic = ['D5+', 'D5', 'D4', 'D3', 'D2', 'D1', 'U1', 'U2', 'U3', 'U4', 'U5', 'U5+']
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(dic)):
        if x == dic[i]:
            result[i] = 1
            return i
    # print('default category')
    return 5

def process_result(experiment_result):
    column_set = ['Market Share', 'Company Strategies', 'Products Performance', 'Industry Status',
                   'Investor Sentiment', 'Stock Risk', 'Competitor Status', 'Supplier Status',
                   'Innovation Sustainability']
    for i in range(len(column_set)):
        if i == len(column_set)-1:
            experiment_result[column_set[i] + '_reply'] = experiment_result['result'].apply(lambda x: category(x, column_set[i],'end_token_that_never_meet'))
            experiment_result[column_set[i] + '_digit'] = experiment_result[column_set[i] + '_reply'].apply(lambda x: generating_digit(x))
        else:
            experiment_result[column_set[i] + '_reply'] = experiment_result['result'].apply(lambda x: category(x, column_set[i],column_set[i+1]))
            experiment_result[column_set[i] + '_digit'] = experiment_result[column_set[i] + '_reply'].apply(lambda x: generating_digit(x))

    return experiment_result

def process_result_onehot(experiment_result):
    column_set = ['Market Share', 'Company Strategies', 'Products Performance', 'Industry Status',
                   'Investor Sentiment', 'Stock Risk', 'Competitor Status', 'Supplier Status',
                   'Innovation Sustainability']
    for col in column_set:
        if col == column_set[-1]:
            experiment_result[col + '_reply'] = experiment_result['result'].apply(lambda x: category(x, col, 'end_token_that_never_meet'))
        else:
            experiment_result[col + '_reply'] = experiment_result['result'].apply(lambda x: category(x, col, column_set[column_set.index(col)+1]))
        experiment_result[col + '_onehot'] = experiment_result[col + '_reply'].apply(lambda x: generating_onehot(x))
        experiment_result[[f"{col}_pos", f"{col}_neg", f"{col}_neu"]] = pd.DataFrame(experiment_result[col + '_onehot'].tolist(), index=experiment_result.index)
    return experiment_result


def assign_ground_truth(exp_result, ts_2024):
    # ts_2024['Date'] = ts_2024['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
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

if __name__ == "__main__":
    mode = 'digit'  # 'onehot' or 'digit'
    # Load the dataset
    file_name = "output/eng/HF.parquet"
    exp_result = pd.read_parquet(file_name)
    if 'chn' in file_name:
        flag = 'CHN'
    else:
        flag = 'USA'

    if flag == 'CHN':
        ts_2024 = pd.read_pickle('data/csi100_ts_2024.pkl')
    else:
        ts_2024 = pd.read_pickle('data/nasdaq_ts_2024.pkl')
    if mode == 'onehot':
        processed = process_result_onehot(exp_result)
    else:
        processed = process_result(exp_result)

    if flag == 'CHN':
        ts_2024['Date'] = ts_2024['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    processed = assign_ground_truth(processed, ts_2024)
    experiment_result = processed.set_index(['date', 'symbol'])
    date_list = list(set([x[0] for x in experiment_result.index.tolist()]))
    experiment_result = experiment_result.reset_index()
    date_list.sort()
    if mode == 'onehot':
        feature_columns = []
        for col in ['Market Share', 'Company Strategies', 'Products Performance', 'Industry Status','Investor Sentiment', 'Stock Risk', 'Competitor Status', 'Supplier Status', 'Innovation Sustainability']:
            feature_columns += [f"{col}_pos", f"{col}_neg", f"{col}_neu"]
    else:
        feature_columns = ['Market Share_digit', 'Company Strategies_digit', 'Products Performance_digit',
                        'Industry Status_digit', 'Investor Sentiment_digit', 'Stock Risk_digit',
                        'Competitor Status_digit', 'Supplier Status_digit', 'Innovation Sustainability_digit']
    experiment_result['mc_gt'] = experiment_result['ground_truth'].apply(lambda x: helper(x))
    gt_list = []
    pred = []
    model = RandomForestClassifier(max_depth=2, random_state=42)


    lookback = 1
    for i in tqdm(range(lookback, len(date_list))):
        x_train = experiment_result[experiment_result['date'] == date_list[i - 1]][feature_columns]
        y_train = experiment_result[experiment_result['date'] == date_list[i - 1]]['mc_gt']
        # x_train = experiment_result[experiment_result['date'].isin(date_list[i-lookback:i])][feature_columns]
        # y_train = experiment_result[experiment_result['date'].isin(date_list[i-lookback:i])]['mc_gt'].tolist()
        model.fit(x_train, y_train)  # only fit one-day result
        x_test = experiment_result[experiment_result['date'] == date_list[i]][feature_columns]
        current_gt = experiment_result[experiment_result['date'] == date_list[i]]['mc_gt'].tolist()
        gt_list += current_gt
        predictions = model.predict(x_test).tolist()
        pred += predictions
        experiment_result.loc[experiment_result['date'] == date_list[i], 'mc_pred'] = predictions

    accuracy = accuracy_score(gt_list, pred)
    f1 = f1_score(gt_list, pred, average='weighted')
    print("Multi-class Accuracy:", accuracy)
    print("Multi-class F1 Score:", f1)

    if mode == 'onehot':
        if flag == 'CHN':
            preds_name = 'preds/chn/' + file_name.split('/')[-1].replace('.parquet', '_preds_onehot.csv')
        else:
            preds_name = 'preds/eng/' + file_name.split('/')[-1].replace('.parquet', '_preds_onehot.csv')
    else:
        if flag == 'CHN':
            preds_name = 'preds/chn/' + file_name.split('/')[-1].replace('.parquet', '_preds.csv')
        else:
            preds_name = 'preds/eng/' + file_name.split('/')[-1].replace('.parquet', '_preds.csv')
    preds_file = experiment_result[['date', 'symbol', 'mc_gt', 'mc_pred']]
    preds_file.to_csv(preds_name, index=False)