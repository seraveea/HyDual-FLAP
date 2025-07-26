import pandas as pd
import transformers
import torch
from tqdm import tqdm
import argparse


def ll3_instance(args):
    model_dir = "/data/user/seraveea/research/hugging_face_cache/Meta-Llama-3-8B-Instruct_hf"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    return pipeline


def build_prompt(line):
    return f"""Title: {line['Title']}\nDescription: {line['Description']}\nLong Description: {line['Long description']}
    """

def build_prompt_zh(line):
    return f"""标题: {line['title']}\n内容: {line['content']}
    """



def summary_tool(client, text):
    """
    given a llm client, a single document, ask llm to classify it to a tuple or event.
    """
    messages = [
        {"role": "system", "content":
            """
            Help me summarize the given news.  
            Please list at least two entities that are unique nouns after the summarization under "Key Entities:".
            """
         },
        {"role": "user", "content": text}
    ]
    # get the reply from llama3 agent
    prompt = client.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        client.tokenizer.eos_token_id,
        client.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = client(prompt,max_new_tokens=200,eos_token_id=terminators,do_sample=True,temperature=0.01,top_p=0.9,pad_token_id=client.tokenizer.eos_token_id)
    result = outputs[0]["generated_text"][len(prompt):]
    return result

def summary_tool_zh(client, text):
    """
    given a llm client, a single document, ask llm to classify it to a tuple or event.
    """
    messages = [
        {"role": "system", "content":
            """
            帮我用中文总结下列新闻
            请在总结后在新的一行内列出至少两个独特的名词实体，作为“关键实体”。
            """
         },
        {"role": "user", "content": text}
    ]
    # get the reply from llama3 agent
    prompt = client.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        client.tokenizer.eos_token_id,
        client.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = client(prompt,max_new_tokens=400,eos_token_id=terminators,do_sample=True,temperature=0.01,top_p=0.9,pad_token_id=client.tokenizer.eos_token_id)
    result = outputs[0]["generated_text"][len(prompt):]
    return result



def main(args):
    torch.compile(mode='reduce-overhead')  # try to speed up the llama3
    text_df = pd.read_pickle(args.text_path)
    llm = ll3_instance(args)
    initial_dict = {
        'published time': [],
        'symbol': [],
        'llm_reply': [],
        'url': []
    }
    res_df = pd.DataFrame(initial_dict)
    # Filter text_df by date range
    if args.language == 'zh':
        date_col = 'date'
    else:
        date_col = 'Published time'
    text_df = text_df[(text_df[date_col] >= args.start_date) & (text_df[date_col] < args.end_date)].reset_index(drop=True)
    for i in tqdm(range(text_df.shape[0])):
        if args.language == 'en':
            prompt = build_prompt(text_df.iloc[i])
            result = summary_tool(llm, prompt)
            res_dict = {
            'published time': [text_df.iloc[i]['Published time']],
            'symbol': [text_df.iloc[i]['Symbol']],
            'llm_reply': [result],
            'url': [text_df.iloc[i]['URL']]
            }
        else:
            prompt = build_prompt_zh(text_df.iloc[i])
            result = summary_tool_zh(llm, prompt)
            res_dict = {
            'published time': [text_df.iloc[i]['date']],
            'llm_reply': [result],
            'symbol': [text_df.iloc[i]['Company Symbol']],
            'url': [text_df.iloc[i]['url']]
            }


        temp_df = pd.DataFrame(res_dict)
        res_df = pd.concat([res_df, temp_df], ignore_index=True)
        res_df.reset_index()
    res_df.to_pickle(args.result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='data/new_llama3/csi100_summary_Nov.pkl')
    parser.add_argument('--text_path', default='data/raw_chn_news/csi100_2stquarter_news.pkl')
    parser.add_argument('--start_date', default='20231201', help='[start date, end date)')
    parser.add_argument('--end_date', default='20231231')
    parser.add_argument('--language', default='zh', choices=['zh', 'en'])
    parser.add_argument('--device', default='cuda:0')
    the_args = parser.parse_args()
    return the_args


if __name__ == '__main__':
    my_args = parse_args()
    main(my_args)

