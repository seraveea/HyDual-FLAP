# HyDual-FLAP Implementation


<!-- The NASDAQ100 and CSI100 news documents will be released after the double-blind review process. -->

## Quick Evaluation
To quickly evaluate the performance of our methods and baselines, we provide all the necessary scripts and data. You can verify the results in a few minutes by following the steps below.

In case you don't want to run any experiments, we provide the backtest result csv file in ```preds``` folder.

### Environment Setup
Please make sure that you have following packages installed:
```bash
sklearn;pandas;tqdm;numpy
```
This is the minimum requirement to run the evaluation scripts.
### Multiclass Classification
We provide all experiments result in output/eng and output/chn folders. 
To evaluate the multiclass classification performance, you can run the following command:
```
python scripts/direct_inference.py
```
For HyDual-FLAP results, you can run:
```
python scripts/indirect_inference.py
```
### Financial Evaluation
After running the above command, you can evaluate the financial performance of our methods and baselines by running:
```
python scripts/backtest.py
```
Then a csv file will be generated in the ```preds``` folder, which contains the financial performance metrics.

## Run experiments
To reproduce the results, please follow the steps below.
### Environment Setup
Please create an environment and install the required packages, we recommend using conda for package management:
```
conda create -n hydual-flap python=3.12
conda activate hydual-flap
pip install -r requirements.txt
```

### Data/Model Preparation
1. Download the LLMs from huggingface
2. Download the financial news dataset, we will release the NASDAQ100 and CSI100 news documents after the double-blind review process.

### Run experiments


#### NASDAQ100 
##### HyDual-FLAP
```
python scripts/graph_exp.py --first_trading_day '2024-01-01' --last_trading_day '2024-03-01' --result_path '[your result path]' --runtime_recording_path '[your log path]' --style 'indirect' --model_name 'temp_walk'
```
##### Baselines without Retrieval
```
python scripts/nograph_exp.py --result_path '[your result path]' --runtime_recording_path '[your log path]' --style 'direct' --model_name 'GCS' --backbone '[llama3/GPT/FinPTForecaster/deepseek]'
```
##### Baselines with Retrieval
Naive RAG on [llama3/GPT/FinPTForecaster/deepseek]
```
python scripts/nograph_exp.py --result_path '[your result path]' --runtime_recording_path '[your log path]' --style 'direct' --model_name 'RAG' --backbone '[llama3/GPT/FinPTForecaster/deepseek]'
```
ChatQA
```
python scripts/nograph_exp.py --result_path '[your result path]' --runtime_recording_path '[your log path]' --style 'direct' --model_name 'RAG' --backbone 'chatqa'
```
Tempralm
```
python scripts/nograph_exp.py --result_path '[your result path]' --runtime_recording_path '[your log path]' --style 'direct' --model_name 'TempRALM'
```

#### CSI100
For CSI100 experiments, please add extra parameters as follows:
```
--source_path 'data/news_llama3/csi100_summary.pkl' 
--ts_path 'data/csi100_ts_2024.pkl'
--summary_path 'data/csi100_business_summary.pkl'
```
Another parts are the same as NASDAQ100 experiments.