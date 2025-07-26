import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
                        
def select_stocks_with_random_tiebreak(df, top_k, seed=42):
    np.random.seed(seed)
    df = df.copy()
    df = df.groupby('MP', group_keys=False).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    df = df.sort_values(by='MP', ascending=False)
    return df.head(top_k)


def backtest(result, stock_list):
    result['Date'] = pd.to_datetime(result['Date'])
    all_dates = result['Date'].sort_values().unique()
    portfolio_returns = []

    for d in all_dates:
        day_df = result[result['Date'] == d]
        top_selected = select_stocks_with_random_tiebreak(day_df, top_k=10, seed=42)
        portfolio_return = top_selected['daily_return'].mean() if not top_selected.empty else 0

        portfolio_returns.append({
            'date': d,
            'portfolio_return': portfolio_return
        })

    portfolio_df = pd.DataFrame(portfolio_returns)
    portfolio_df['cumulative_return'] = (1 + portfolio_df['portfolio_return']).cumprod()
    portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
    total_days = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
    annualized_return = portfolio_df['cumulative_return'].iloc[-1] ** (252 / total_days) - 1
    annualized_volatility = portfolio_df['portfolio_return'].std() * np.sqrt(252)

    sharpe_ratio = annualized_return / annualized_volatility

    portfolio_df['cum_max'] = portfolio_df['cumulative_return'].cummax()
    portfolio_df['drawdown'] = portfolio_df['cumulative_return'] / portfolio_df['cum_max'] - 1
    max_drawdown = portfolio_df['drawdown'].min()

    calmar_ratio = annualized_return / abs(max_drawdown)


    return {"sr": sharpe_ratio, "cr": calmar_ratio, "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility, "max_drawdown": max_drawdown}

def main(flag, file_name):
    if flag == 'CHN':
        ts = pd.read_pickle('data/csi100_ts_2024.pkl')
        # print(ts.head(5))
        csi_dict = pd.read_csv('data/index_csi100.csv')
        stock_list = csi_dict['symbol'].tolist()
        stock_list = [s.strip() for s in stock_list]
    else:
        ts = pd.read_pickle('data/nasdaq_ts_2024.pkl')
        nas_dict = pd.read_csv('data/index_nasdaq.csv')
        stock_list = nas_dict['symbol'].tolist()

    rr = ts[['Date','Symbol','daily_return']]
    pred = pd.read_csv(file_name)
    pred = pred.rename(columns={'mc_pred':'MP','date':'Date', 'symbol':'Symbol','mc_gt':'MGT'})
    ######## only for chn ########
    if flag == 'CHN':
        pred['Symbol'] = pred['Symbol'].astype(str).str.zfill(6)
        rr['Date'] = pd.to_datetime(rr['Date']).dt.strftime('%Y-%m-%d')
    ##############################

    result = pd.merge(rr,pred,on=['Date','Symbol'],how='inner')
    return backtest(result, stock_list)


def get_all_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# change to CHN for csi100
flag = 'CHN'

if flag == 'CHN':
    all_files = get_all_csv_files('preds/chn')
    result = []
    for i in tqdm(all_files):
        col = i.replace('preds/chn/', '').replace('_preds.csv', '')
        temp = main('CHN', 'preds/chn/'+i)
        temp['index'] = col
        result.append(temp)
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by='sr', ascending=False)
    print(result_df)
    result_df.to_csv('preds/chn_backtest_summary.csv', index=False)
else:
    all_files = get_all_csv_files('preds/eng')
    result = []
    for i in tqdm(all_files):
        col = i.replace('preds/eng/', '').replace('_preds.csv', '')
        temp = main('ENG', 'preds/eng/'+i)
        temp['index'] = col
        result.append(temp)
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by='sr', ascending=False)
    print(result_df)
    result_df.to_csv('preds/eng_backtest_summary.csv', index=False)


