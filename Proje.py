import numpy as np
import pandas as pd
import pandas_datareader.data as web
import random
import fix_yahoo_finance as yf
 
 
#list of stocks in portfolio(For different stocks, use finance.yahoo.com to find stock names. If you want to add more stocks, please look at line 36.
stocks = sorted(['ULKER.IS','TUPRS.IS','AKBNK.IS','TCELL.IS'])
data = web.DataReader(stocks,data_source='yahoo',start='01/01/2010')['Adj Close']
 #convert daily stock prices into daily returns
returns = data.pct_change()
#calculate mean daily return and covariance of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 2500

results = np.zeros((4+len(stocks)-1,num_portfolios))
 
for i in range(num_portfolios):
    #random weights
    weights = np.array(np.random.uniform(-0.3,0.3,4))
    weights /= np.sum(weights)
    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility)
    results[2,i] = results[0,i] / results[1,i]
    
    for j in range(len(weights)):
        results[j+3,i] = weights[j]
 

results_frame = pd.DataFrame(results.T,columns=['Return','Risk','Sharpe',stocks[0],stocks[1],stocks[2],stocks[3]]) #Number of "stocks[n]" should be equal to number of actual stocks.
 
#Portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['Sharpe'].idxmax()]
#Portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['Risk'].idxmin()]

 
print("Results with maximum Sharpe Ratio:",max_sharpe_port)
print("Results with minimum risk:",min_vol_port)
