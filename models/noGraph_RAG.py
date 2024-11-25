import requests
import json
import sys
import time
sys.path.insert(0, sys.path[0] + "/../")
from scripts.utils import *

"""
Notes: noGraph RAG agents are implemented here
"""


class vanilla_RAG:
    def __init__(self, client, time_series, business_summary, maper, embeder, preload_doc_path, style):
        """
        :param client: llama3 instance
        :param time_series: a df with all time series and ground truth
        :param maper: a dictionary that contain the company full name and symbol
        :param embeder: a bert embeder
        """
        self.client = client
        self.time_series = time_series
        self.business_summary = business_summary
        self.maper = dict(zip(maper['symbol'], maper['company full name']))
        self.embeder = embeder
        self.style = style
        self.run_times = []
        self.inference_times = []
        if preload_doc_path:
            self.preload_doc = pd.read_pickle(preload_doc_path)
            self.preload_flag = True
        else:
            self.preload_doc = None
            self.preload_flag = False

    def rag_reply(self, symbol, date, sub_dataset, topk, backbone):
        """
        :param backbone:
        :param symbol: the stock symbol we ask
        :param date: the trading day we ask
        :param sub_dataset: a hugging_face dataset with recent documents
        :param topk: number of retrieved documents
        :return: one result df and one df recording which file retrieved
        """
        # **********
        start_time = time.time()
        # **********
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                docs = sub_dataset.filter(lambda e: self.retrieve_based_on_given_list(e, doc_list))
                separate_dict = separate_dictionary(docs.to_dict(), topk)
            except:
                print('c1')
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                        'N/A',
                                        [{'url': 'empty'}],
                                        0)
        else:
            query = self.default_query(symbol, date)
            q_vector = np.array(self.embeder.embed(query))
            scores, retrieved_doc = sub_dataset.get_nearest_examples('embedding', q_vector, k=topk)
            separate_dict = separate_dictionary(retrieved_doc, topk, scores)
        time_series, ground_truth = self.find_ground_truth(symbol, date)

        retrieval_prompt = generate_prompt_ll3(separate_dict, date, symbol, time_series)

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

    def gcs_reply(self, symbol, date, sub_dataset, topk, backbone):
        # **********
        start_time = time.time()
        # **********
        if self.preload_flag:
            line = self.preload_doc[(self.preload_doc['symbol'] == symbol) &
                                    (self.preload_doc['date'] == date)]['retrieved files']
            try:
                assert line.shape[0] == 1
                doc_list = line.values[0]
                docs = sub_dataset.filter(lambda e: self.retrieve_based_on_given_list(e, doc_list))
                separate_dict = separate_dictionary(docs.to_dict(), topk)
            except:
                return self.get_dict(symbol, date, 'ERROR: No preload files',
                                     'N/A',
                                     [{'url': 'empty'}],
                                     0)
        else:
            doc_dataset = sub_dataset.filter(lambda x: (x['symbol'] == symbol))
            sort_doc = doc_dataset.sort('published time')
            if len(sort_doc) < topk:
                separate_dict = separate_dictionary(sort_doc[:], len(sort_doc))
            else:
                separate_dict = separate_dictionary(sort_doc[-1 * topk:], topk)
        time_series, ground_truth = self.find_ground_truth(symbol, date)
        retrieval_prompt = generate_prompt_ll3(separate_dict, date, symbol, time_series)

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

    def client_reply(self, retrieval_prompt, symbol, date, ground_truth, separate_dict):
        start_time = time.time()
        # return a df and a dict
        message = self.initialize_message(retrieval_prompt, symbol)
        prompt = self.set_prompt(message)
        prompt_len = len(prompt)
        # need more tokens for CoT inference
        output = self.client(prompt, max_new_tokens=1000,
                             eos_token_id=[self.client.tokenizer.eos_token_id,
                                           self.client.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                             do_sample=True,
                             temperature=0.1, top_p=0.9, pad_token_id=self.client.tokenizer.eos_token_id)
        result = output[0]["generated_text"][len(prompt):]
        end_time = time.time()
        run_time = end_time - start_time
        self.inference_times.append(run_time)
        return self.get_dict(symbol, date, result, ground_truth, separate_dict, prompt_len)

    def gpt_reply(self, retrieval_prompt, symbol, date, ground_truth, separate_dict):
        """
        new client reply for gpt-4o
        return a df and a dict
        """
        message = self.initialize_message(retrieval_prompt, symbol)
        message.append({"role": "user", "content": retrieval_prompt})
        url = 'please replace with your prefer GPT service provider'
        headers = {
            "and write this part according to the provider's requirement"
        }
        data = {
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            result = response.json()['choices'][0]['message']['content']
            prompt_len = 'N/A'
        except requests.exceptions.Timeout:
            print('timeout error')
            result = 'timeout error'
            prompt_len = 'N/A'
        except Exception as e:
            print('api error')
            result = 'api error'
            prompt_len = 'N/A'

        return self.get_dict(symbol, date, result, ground_truth, separate_dict, prompt_len)

    def find_ground_truth(self, symbol, date):
        ts_dataset = self.time_series[(self.time_series['Date'] == date) & (self.time_series['Symbol'] == symbol)]
        # SPLK close at Mar 15 2024
        if ts_dataset.shape[0] != 1:
            print('stock series not found or conflict')
            time_series = 'Please ignore this part'
            ground_truth = 'N/A'
        else:
            time_series = 'placeholder'
            ground_truth = ts_dataset['daily_return_bins'].tolist()[0]

        return time_series, ground_truth

    def initialize_message(self, retrieval_prompt, symbol):
        symbol_summary = self.business_summary.loc[(0, symbol)]['Summary']
        if self.style == 'direct':
            return [
                {"role": "system", "content":
                    """
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
                    """},
                {"role": "system", "content":
                    """
                    You need to forecast next trading day stock return for stock on date of user given. 
                    The next day stock return is represented by bins "D5+", "D5", "D4", "D3", "D2", "D1", "U1", "U2", "U3", "U4", "U5", "U5+", 
                    where "D5+" means price dropping more than 5%, D5 means price dropping between 4% and 5%, 
                    "D4" means price dropping between 3% and 4%, "U5+" means price rising more than 5%, 
                    "U5" means price rising between 4% and 5%, "D4" means price rising between 3% and 4%, etc.
                    """
                 },
                {"role": "system", "content":
                    """
                    You will given a series of news with released date and content. Released date is placed between <date> and </date>. Content is placed between <doc> and </doc>. 
                    Please describe what you get from news, summary positive and negative factors about the stock prices. 
                    Based on your observation, give the next trading day stock return the format of bins above between $$ and $$ at the last line.
                    """
                 },
                {"role": "user",
                 "content": f"""<Company Summary>{symbol_summary}</Company Summary>\n{retrieval_prompt}"""}
            ]
        elif self.style == 'indirect':
            return [
                {"role": "system", "content":
                    """
                    You will given a series of news with released date and content.
                    Please evaluate given stock's situations ONLY based on the provided news from given perspectives:
                    1. market share: the percentage of total sales a company captures within its industry.
                    2. company strategies: long-term plans that guide business in future development.
                    3. products performance: the quality and profitability of company product or services.
                    4. industry status: the overall marco situation of the company's sector or industry.
                    5. investor sentiment: the overall attitude or mood of investors toward the company.
                    6. stock risk: potential for financial loss due to market volatility, economic changes, or company performance.
                    7. competitor status: the current position, strategies, strengths, and weaknesses of rival companies within the industry.
                    8. supplier status: performance, reliability, capacity, and financial health of the company's suppliers, impacting the supply chain.
                    9. innovation sustainability: the ability to maintain and develop new, impactful innovations over time.
                    Analyze based on below instruction and choose ONLY ONE from [positive, neutral, negative] for each perspective.
                    """
                 },
                {
                    "role": "system", "content":
                    """
                    [instruction] When you making decisions, follow the steps:
                    -Step 1: recognize each document is about the company itself, or about the industry, competitors or suppliers.
                    -Step 2: Examine every document content and category it to different perspectives.
                    -Step 3: In every perspective, combine all corresponding documents and sort them by date, more recent one have more weight, and choose from [positive, neutral, negative].
                    """
                },
                {"role": "user",
                 "content": f"""{retrieval_prompt}"""}
            ]

    def set_prompt(self, messages):
        return self.client.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def default_query(self, symbol, date):
        symbol_summary = self.business_summary.loc[(0, symbol)]['Summary']
        if self.style == 'direct':
            return f"""I need estimate the stock price at {date} according to the up-to-date news 
            and information of company {self.maper[symbol]} before {date}. Here is the short business summary:
            {symbol_summary}
            """
        elif self.style == 'indirect':
            return f"""I need evaluate the company {self.maper[symbol]} at {date} from market share, company 
            strategies, products performance, industry status, investor sentiment, 
            stock risk, competitor status, supplier status, innovation sustainability, 
            according to the up-to-date news and information before {date}. Here is the short business summary:
            {symbol_summary}"""

    def log_time(self, log_file='run_times.log'):
        if self.run_times:
            average_time = sum(self.run_times) / len(self.run_times)
            with open(log_file, 'a') as f:
                f.write(f"Average run time: {average_time:.6f} seconds\n")
            print(f"Average run time saved to {log_file}")
        else:
            print("No run times recorded.")

    def log_infer_time(self, log_file='run_times.log'):
        if self.inference_times:
            average_time = sum(self.inference_times) / len(self.inference_times)
            with open(log_file, 'a') as f:
                f.write(f"Average inference time: {average_time:.6f} seconds\n")
            print(f"Average run time saved to {log_file}")
        else:
            print("No run times recorded.")

    @staticmethod
    def get_dict(symbol, date, result, ground_truth, separate_dict, prompt_len):
        result_dict = {
            'symbol': symbol,
            'date': date,
            'result': result,
            'ground_truth': ground_truth,
            'prompt_len': prompt_len
        }
        result_df = pd.DataFrame(result_dict, index=[0])
        retrieved_dict = {
            'symbol': symbol,
            'date': date,
            'retrieved files': [t['url'] for t in separate_dict]
        }
        return result_df, retrieved_dict

    @staticmethod
    def only_return_retrieved_files(separate_dict, symbol, date, ground_truth):
        retrieved_dict = {
            'symbol': symbol,
            'date': date,
            'retrieved files': [t['url'] for t in separate_dict],
            'ground_truth': ground_truth
        }
        return {}, retrieved_dict

    @staticmethod
    def Tempralm_index(result_dict, scores, date, topk):
        timestamp = [x['published time'] for x in result_dict]
        mean_score = np.mean(scores)
        variance_score = np.var(scores)
        tau = [compute_tau(date, x) for x in timestamp]
        mean_tau = np.mean(tau)
        variance_tau = np.var(tau)
        final_tau = [((t - mean_tau) * variance_score) / variance_tau + mean_score for t in tau]
        temp_score = [final_tau[i] + scores[i] for i in range(len(scores))]
        for i in range(len(result_dict)):
            result_dict[i]['temp_score'] = temp_score[i]

        return sorted(range(len(temp_score)), key=lambda i: temp_score[i], reverse=True)[:topk]

    @staticmethod
    def retrieve_based_on_given_list(sub_data, url_list):
        return sub_data['url'] in url_list
