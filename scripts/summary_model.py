import pandas as pd
import transformers
import torch
from tqdm import tqdm
import argparse


def ll3_instance(args):
    model_dir = "your llama3 location"
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
    # that referring to a specific company, industry, item, person or concept,
    terminators = [
        client.tokenizer.eos_token_id,
        client.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = client(
        prompt,
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
        pad_token_id=client.tokenizer.eos_token_id
    )
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
    for i in tqdm(range(text_df.shape[0])):
        prompt = build_prompt(text_df.iloc[i])
        result = summary_tool(llm, prompt)
        res_dict = {
            'published time': [text_df.iloc[i]['Published time']],
            'symbol': [text_df.iloc[i]['Symbol']],
            'llm_reply': [result],
            'url': [text_df.iloc[i]['URL']]
        }
        temp_df = pd.DataFrame(res_dict)
        res_df = pd.concat([res_df, temp_df], ignore_index=True)
        res_df.reset_index()
    res_df.to_pickle(args.result_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='data/new_llama3/nasdaq_summary_Nov23.pkl')
    parser.add_argument('--text_path', default='data/raw_news24/nasdaq_Nov23.pkl')
    parser.add_argument('--device', default='auto')

    the_args = parser.parse_args()

    return the_args


if __name__ == '__main__':
    my_args = parse_args()
    main(my_args)

