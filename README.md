# Stock-Prediction-Meets-LLM-Knowledge-Graph-Augmented-Generation-with-Bias-free-Inference


Please download the news and knowledge data from hugging face https://huggingface.co/datasets/seraveea/NASDAQ100_knowledge_graph_in_2024_Jan2Jun
```
# Our methods
python scripts/graph_exp.py --result_path 'output/reply/temp_walk.pkl' --style 'indirect'

# Baselines: LLaMA3+GCS
python scripts/nograph_exp.py --result_path 'output/reply/llama3_gcs.pkl' --backbone 'llama3' --model_name GCS --style 'direct'

# Baselines: LLaMA3+DPR
python scripts/nograph_exp.py --result_path 'output/reply/llama3_rag.pkl' --backbone 'llama3' --model_name RAG --style 'direct'

# Baselines: GPT4+GCS
python scripts/nograph_exp.py --result_path 'output/reply/gpt4_gcs.pkl' --backbone 'gpt4o' --model_name GCS --style 'direct'

# Baselines: GPT4+DPR
python scripts/nograph_exp.py --result_path 'output/reply/gpt4_rag.pkl' --backbone 'gpt4o' --model_name RAG --style 'direct'

# Baselines: FinGPT+GCS
python scripts/nograph_exp.py --result_path 'output/reply/fingpt_gcs.pkl' --backbone 'fingpt' --model_name GCS --style 'direct'

# Baselines: FinGPT+DPR
python scripts/nograph_exp.py --result_path 'output/reply/fingpt_rag.pkl' --backbone 'fingpt' --model_name RAG --style 'direct'
```
