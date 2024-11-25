from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random
import torch
from dateutil.relativedelta import relativedelta
from transformers import BertTokenizer, BertModel
from torch import tensor


def x_month_ago(date_str, month_num=1):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    x_month_ago_date = date_obj - relativedelta(months=month_num)

    return x_month_ago_date.strftime("%Y-%m-%d")


class Bert_embeder:
    def __init__(self):
        random_seed = 42
        random.seed(random_seed)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='hugging_face_cache')
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir='hugging_face_cache')

    def embed(self, text):
        encoding = self.tokenizer.batch_encode_plus(
            text,  # List of input texts
            padding=True,  # Pad to the maximum sequence length
            truncation=True,  # Truncate to the maximum sequence length if necessary
            return_tensors='pt',  # Return PyTorch tensors
            add_special_tokens=True  # Add special tokens CLS and SEP
        )

        input_ids = encoding['input_ids']  # Token IDs
        # print input IDs
        attention_mask = encoding['attention_mask']  # Attention mask
        # print attention mask
        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # This contains the embeddings
            word_embeddings = word_embeddings.mean(dim=0)  # -> 3x768
            word_embeddings = word_embeddings.mean(dim=0)  # -> 768

        return word_embeddings


def map_vector(x):
    x['vector'] = np.array(x['vector'])
    return x


def format_date(x):
    date_obj = datetime.strptime(x['published time'], "%Y%m%d")
    x['published time'] = date_obj.strftime("%Y-%m-%d")
    return x


def pd_format_date(x):
    date_obj = datetime.strptime(x, "%Y%m%d")
    x = date_obj.strftime("%Y-%m-%d")
    return x


def map_pos_vector(x):
    embedding = eval(x['pos_vector'])
    if len(embedding) > 1:
        x['pos_vector'] = np.array(embedding)
    else:
        x['pos_vector'] = np.array(embedding[0])
    return x


def separate_dictionary(original_dict, num_values, addition_value=None):
    # if the candidated number is lower than num_values
    if len(original_dict['published time']) < num_values:
        num_values = len(original_dict['published time'])

    # Initialize empty dictionaries
    separated_dicts = [{} for _ in range(num_values)]

    # Iterate over the keys of the original dictionary
    for key in original_dict:
        # Iterate over the number of values specified
        for i in range(num_values):
            # Assign one value to each new dictionary
            separated_dicts[i][key] = original_dict[key][i]

    if addition_value is not None:
        for i in range(num_values):
            separated_dicts[i]['relevant score'] = addition_value[i]

    return separated_dicts


def analyze_doc(doc_list):
    # this function is used to generate the content of all documents
    content = []
    for doc in doc_list:
        prefix = 'date: ' + str(doc['Date'])
        if doc['Type'] == 'Summary':
            content.append(prefix + ' [static knowledge] ' + doc['Summary'])
        elif doc['Type'] == 'News':
            content.append(prefix + ' [news] about ' + deal_with_nan(doc['Symbol']) + 'Title: '
                           + deal_with_nan(doc['Title']) + 'Description: ' + deal_with_nan(doc['Description'])
                           + 'Long Description: ' + deal_with_nan(doc['Long description']))
        else:
            symbol = doc['Symbol']
            date = doc['Date']
            content.append(f'time series of stock {symbol} in day {date}: ' + doc['series'])
    return content


def get_trading_days(start_date, end_date):
    # Load the US stock market calendar
    us_calendar = get_calendar('XNYS')

    # Generate the trading days between the specified start and end dates
    trading_days = us_calendar.schedule(start_date=start_date, end_date=end_date)

    # Extract the trading days and format them as strings in the desired format
    trading_day_strings = [day.strftime('%Y-%m-%d') for day in trading_days.index]

    return trading_day_strings


def get_first_and_last_trading_days(start_date, end_date):
    nyse = get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

    trading_days = pd.to_datetime(trading_days)

    trading_days_df = pd.DataFrame(trading_days, columns=['date'])
    trading_days_df['year'] = trading_days_df['date'].dt.year
    trading_days_df['week'] = trading_days_df['date'].dt.strftime('%Y-%W')

    grouped = trading_days_df.groupby('week')['date']
    first_days = grouped.first()
    last_days = grouped.last()

    first_days_str = first_days.dt.strftime('%Y-%m-%d').tolist()
    last_days_str = last_days.dt.strftime('%Y-%m-%d').tolist()

    return first_days_str, last_days_str


def compute_tau(query_time, doc_time, alpha=1):
    time_1 = datetime.strptime(query_time, '%Y-%m-%d')
    time_2 = datetime.strptime(doc_time, '%Y-%m-%d')
    return alpha / ((time_1 - time_2).days + 1)


def cosine_similarity(vector1, vector2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    vector1: numpy array or list, representing the first vector
    vector2: numpy array or list, representing the second vector

    Returns:
    similarity: float, representing the cosine similarity between the two vectors
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def deal_with_nan(x):
    if x is None:
        return ''
    else:
        return x


def positional_encoding(position, d_model):
    position_encoding = np.zeros((position, d_model))

    for pos in range(position):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * i) / np.float32(d_model))
            position_encoding[pos, i] = np.sin(angle)
            position_encoding[pos, i + 1] = np.cos(angle)

    return position_encoding


def log_positional_encoding(position, d_model):
    position_encoding = np.zeros((position, d_model))

    for pos in range(position):
        nonlinear_pos = np.log(pos + 1)
        for i in range(0, d_model, 2):
            angle = nonlinear_pos / np.power(100, (2 * i) / np.float32(d_model))
            position_encoding[pos, i] = np.sin(angle)
            position_encoding[pos, i + 1] = np.cos(angle)

    return position_encoding


def map_dates_to_values(start_date, end_date, d_model):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    num_days = (end_date - start_date).days + 1
    pos_encoding = positional_encoding(num_days, d_model)
    date_to_value = {}
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        date_to_value[current_date.strftime('%Y-%m-%d')] = pos_encoding[i]

    return date_to_value


def map_dates_to_valuesv2(start_date, end_date, d_model):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    num_days = (end_date - start_date).days + 1
    pos_encoding = log_positional_encoding(num_days, d_model)
    date_to_value = {}
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        date_to_value[current_date.strftime('%Y-%m-%d')] = pos_encoding[i]

    ns = date_to_value[end_date.strftime('%Y-%m-%d')]
    l2_distance_dict = {}
    for key, vector in date_to_value.items():
        l2_distance = np.linalg.norm(ns - vector)
        l2_distance_dict[key] = l2_distance
    min_value = min(l2_distance_dict.values())
    max_value = max(l2_distance_dict.values())
    normalized_l2_distance_dict = {
        key: 1 - (value - min_value) / (max_value - min_value)
        for key, value in l2_distance_dict.items()
    }
    # return date_to_value
    return normalized_l2_distance_dict


def generate_prompt_ll3(doc_list, date, symbol, ts):
    if len(doc_list) == 0:
        return ''
    else:
        sentences_list = [f"<Target Company>: {symbol}; <DATE>{date}; <NEWS>:\n"]
        if 'Title' in doc_list[0].keys():
            for docu in doc_list:
                temp = f"""<date>{docu['published time']}</date><doc>Title: {docu['Title']}\n
                Description: {docu['Description']}\nLong Description: {docu['Long description']}</doc>
                """
                sentences_list.append(temp)
        else:
            for docu in doc_list:
                temp = f"<date>{docu['published time']}</date><doc>{deal_with_nan(docu['one_sentence'])}<\doc>\n"
                sentences_list.append(temp)
        # sentences_list.append(f'<TS>{ts}</TS>')
        return ' '.join(sentences_list)


def documents_prompt_ll3(doc_list):
    sentences_list = []
    for docu in doc_list:
        if docu['Type'] == 'News':
            temp = f"<date>{docu['Date']}</date><doc>{deal_with_nan(docu['Title'])} {deal_with_nan(docu['Description'])} {deal_with_nan(docu['Long description'])}<\doc>\n"
            sentences_list.append(temp)
    return '\n'.join(sentences_list)


def argument_keyword(s):
    start_index = s.rfind('/') + 1
    end_index = s.rfind('.pkl')
    if start_index != -1 and end_index != -1:
        return s[start_index:end_index]
    else:
        return None
