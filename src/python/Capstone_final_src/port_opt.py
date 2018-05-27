# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:32:03 2018

@author: Aditya
"""

#import pandas.io.data as web

from pandas_datareader import data as web
import pandas as pd

import numpy as np




stocks = ['GOOGL', 'TM', 'KO', 'AMZN']
numAssets = len(stocks)
source = 'yahoo'
start = '2015-01-01'
end = '2017-10-31'

#Retrieve stock price data and save just the dividend adjusted closing prices

timeseries_date = pd.date_range(start, end)

data = pd.DataFrame(index = timeseries_date)

for symbol in stocks:
        data[symbol] = web.DataReader(symbol, data_source=source, start=start, end=end)['Adj Close']

#Calculate simple linear returns
        
data = data.dropna()

returns = (data - data.shift(1)) / data.shift(1)

#Calculate individual mean returns and covariance between the stocks

meanDailyReturns = returns.mean()
covMatrix = returns.cov()

#Calculate expected portfolio performance

weights = [0.5, 0.2, 0.2, 0.1]
portReturn = np.sum( meanDailyReturns*weights )
portStdDev = np.sqrt(np.dot(weights, np.dot(covMatrix, weights)))


print (returns)

print (portReturn)

print (     portStdDev)




import numpy as np
import scipy.optimize as sco

def calcPortfolioPerf(weights, meanReturns, covMatrix):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''    
    #Calculate return and variance

    portReturn = np.sum( meanReturns*weights )
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))

    return portReturn, portStdDev

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_var

def getPortfolioVol(weights, meanReturns, covMatrix):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]

def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate):
    '''
    Finds the portfolio of assets providing the maximum Sharpe Ratio

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts

def findMinVariancePortfolio(meanReturns, covMatrix):
    '''
    Finds the portfolio of assets providing the lowest volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts

#Find portfolio with maximum Sharpe ratio

maxSharpe = findMaxSharpeRatioPortfolio( meanDailyReturns, covMatrix,
                                        0)

print (maxSharpe)
rp, sdp = calcPortfolioPerf(maxSharpe['x'], meanDailyReturns, covMatrix)

print (rp)

print (sdp)

#Find portfolio with minimum variance

minVar = findMinVariancePortfolio( meanDailyReturns, covMatrix)
rp, sdp = calcPortfolioPerf(minVar['x'], meanDailyReturns, covMatrix)

