
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime
import datetime

get_ipython().magic('matplotlib inline')


class technical_indicators():
    
    def __init__(self):        
        self.stock = None       
        

    #Function for difference between current day's opening price and previous day's closing price
    def pcod(self, stock):        
        self.stock = stock        
        self.stock['pcod'] = self.stock['Open'] - self.stock['Close'].shift(1)
        return self.stock    
    
    
    #Function for difference between current day's highest price and lowest price
    def hld(self, stock):        
        self.stock = stock        
        self.stock['hld'] = self.stock['High'] - self.stock['Low']
        return self.stock    
    
    
    # Function for Relative Strength Index
    def stockRsi(self, stock, numOfDays):         
        self.stock = stock
        self.numOfDays = numOfDays        
        delta = self.stock['Close'].diff()    
        deltaUp = np.where(delta>0, delta, 0)
        deltaDown = np.where(delta<0, delta, 0)    
        dfDeltaUp = pd.DataFrame(data=deltaUp, index = stock.index , columns=['deltaUp'])
        dfDeltaDown = pd.DataFrame(data=deltaDown,index=stock.index ,  columns=['deltaDown'])
        rollingDf = pd.DataFrame(np.zeros, index=stock.index, 
            columns=['rollingUp','rollingDown'])
        rollingDf['rollingUp'] = dfDeltaUp.rolling(window=numOfDays).mean()
        #rollingDf['rollingUp'].fillna(0, inplace=True)
        rollingDf['rollingDown'] = dfDeltaDown.rolling(window=numOfDays).mean()
        #rollingDf['rollingDown'].fillna(0, inplace=True)
        rollingDf['RS'] = (rollingDf['rollingUp'] / rollingDf['rollingDown']).abs()
        rsi = 100.0 - (100.0/(1.0 + rollingDf['RS']))
        self.stock['rsi'] = rsi
        return self.stock    
    
    
    #Function to calculate bollinger bands average, upper and lower bands
    def bbands(self, stock, length, numsd):
        self.stock = stock
        self.length = length
        self.numsd = numsd        
        avg = self.stock['Close'].rolling(window=length).mean()
        sd = self.stock['Close'].rolling(window=length).std()
        upband = avg + (sd*numsd)
        dnband = avg - (sd*numsd)
        self.stock['bbul'], self.stock['bbll'] = np.round(upband, 3), np.round(dnband, 3)
        return self.stock    
    
    
        # Function for short-term moving average
    def stma(self, stock, numOfDays):        
        self.stock = stock
        self.numOfDays = numOfDays        
        self.stock['stma'] = self.stock['Close'].rolling(window=numOfDays).mean()
        return self.stock    
    
    
    # Function for long-term moving average
    def ltma(self, stock, numOfDays):        
        self.stock = stock
        self.numOfDays = numOfDays        
        self.stock['ltma'] = self.stock['Close'].rolling(window=numOfDays).mean()
        return self.stock
    
    
    
    # Function for exponential moving average
    def ema(self, stock, numOfDays):        
        self.stock = stock
        self.numOfDays = numOfDays        
        self.stock['ema'] = self.stock['Close'].ewm(span=numOfDays, adjust=False, min_periods=numOfDays-1).mean()
        return self.stock    
    
    
    # self.stock Crossed Upper-Limit
    def scul(self, stock):        
        self.stock = stock        
        self.stock['scul'] = self.stock.Close > self.stock.bbul
        self.stock['scul'] = self.stock['scul'].astype(int)
        return self.stock    
    
    
    # self.stock Crossed Lower-Limit
    def scll(self, stock):        
        self.stock = stock        
        self.stock['scll'] = self.stock.Close < self.stock.bbll
        self.stock['scll'] = self.stock['scll'].astype(int)
        return self.stock
    
    
    # short-term moving average cuts long-term moving average
    def stma_cuts_ltma(self, stock):        
        self.stock = stock        
        stma = self.stock['stma']
        ltma = self.stock['ltma']
        values1 = [] # stma cuts ltma from below 
        values2 = [] # stma cuts ltma from above 
        
        for k in range(0, len(self.stock)):                
            i = j = k           
            
            if i == 0 :
                values1.append(0)
                values2.append(0)
            else:
                
                if (stma[i] > ltma [j]):
                    
                    if (stma[i-1] < ltma [ j-1]):
                        values2.append(0) # stma cuts ltma from above
                        values1.append(1) # stma cuts ltma from below
                    else:
                        values1.append(0)
                        values2.append(0)
               
                elif (stma[i] <  ltma [j]):
                    
                    if (stma [i  -1 ] > ltma [ j -1]):
                        values2.append(1) # stma cuts ltma from above
                        values1.append(0) # stma cuts ltma from below
                        
                    else:
                        values1.append(0)
                        values2.append(0)
                        
                else:
                    values1.append(0)
                    values2.append(0)
                        
                      # stma cuts ltma from below 
        self.stock['stma_cut_ltma_b'] = values1 

        # stma cuts ltma from above
        self.stock['stma_cut_ltma_a'] = values2
 
        return self.stock
    
    
    # close price cuts short-term moving average
    def close_cuts_stma(self, stock):        
        self.stock = stock         
        close = self.stock['Close']
        stma = self.stock['stma']
        values1 = [] # close price cuts stma from below 
        values2 = [] # close price cuts stma from above 

        for k in range(0, len(self.stock)):                
            i = j = k            
            
            if i == 0 :
                values1.append(0)
                values2.append(0)
            else:
                
                if (close[i] > stma [j]):
                    
                    if (close[i-1] < stma [ j-1]):
                        values2.append(0) # price cuts stma from above
                        values1.append(1) # price cuts stma from below
                    else:
                        values1.append(0)
                        values2.append(0)
               
                elif (close[i] <  stma [j]):
                    
                    if (close [i  -1 ] > stma [ j -1]):
                        values2.append(1) # close cuts stma from above
                        values1.append(0) # close cuts stma from below
                        
                    else:
                        values1.append(0)
                        values2.append(0)
                        
                else:
                    values1.append(0)
                    values2.append(0)   

        # close price cuts stma from below 
        self.stock['close_pr_cut_stma_b'] = values1 

        # close price cuts stma from above
        self.stock['close_pr_cut_stma_a'] = values2

        return self.stock
    
    
    # close price cuts long-term moving average
    def close_cuts_ltma(self, stock):
        self.stock = stock
        close = self.stock['Close']
        ltma = self.stock['ltma']
        values1 = [] # close price cuts ltma from below 
        values2 = [] # close price cuts ltma from above 
        
        for k in range(0, len(self.stock)):                
            i = j = k            
            
            if i == 0 :
                values1.append(0)
                values2.append(0)
            else:
                
                if (close[i] > ltma [j]):
                    
                    if (close[i-1] < ltma [ j-1]):
                        values2.append(0) # price cuts stma from above
                        values1.append(1) # price cuts stma from below
                    else:
                        values1.append(0)
                        values2.append(0)
               
                elif (close[i] <  ltma [j]):
                    
                    if (close [i  -1 ] > ltma [ j -1]):
                        values2.append(1) # close cuts ltma from above
                        values1.append(0) # close cuts ltma from below
                        
                    else:
                        values1.append(0)
                        values2.append(0)
                        
                else:
                    values1.append(0)
                    values2.append(0) 

        # close price cuts ltma from below 
        self.stock['close_pr_cut_ltma_b'] = values1 

        # close price cuts ltma from above
        self.stock['close_pr_cut_ltma_a'] = values2        

        return self.stock    
    
    
    # daily returns(percentage) of the stock
    def daily_returns(self, stock):        
        self.stock = stock        
        price = self.stock['Adj Close']
        price = (price / price.shift(1)) - 1
        self.stock['dr_stock'] = price
        return self.stock
    
    
    # daily returns(percentage) for NSE
    def daily_returns_NSE(self, stock):
        self.stock = stock        
        price = self.stock['NSE']
        price = (price / price.shift(1)) - 1
        self.stock['dr_nse'] = price
        return self.stock    
    
    
    # daily returns(percentage) for SPY
    def daily_returns_SPY(self, stock):        
        self.stock = stock
        price = self.stock['SPY']
        price = (price / price.shift(1)) - 1
        self.stock['dr_spy'] = price
        return self.stock
    
    
    
    # daily returns(percentage) for DOWJONES
    def daily_returns_DOWJONES(self, stock):
        self.stock = stock        
        price = self.stock['DJ']
        price = (price / price.shift(1)) - 1
        self.stock['dr_dj'] = price
        return self.stock
    
    
    
    # daily returns(percentage) for NASDAQ
    def daily_returns_NASDAQ(self, stock):        
        self.stock = stock        
        price = self.stock['NASDAQ']
        price = (price / price.shift(1)) - 1
        self.stock['dr_nasdaq'] = price
        return self.stock    
    
   
    # Price movement (up/down)
    def price_movement(self, stock): # use diff() method of pandas
        self.stock = stock
        close_price = self.stock['Close']
        price_movmnt = []

        for i in range(0, len(self.stock)):

            if i == 0:
                price_movmnt.append(np.nan)

            elif close_price[i] < close_price[i-1]: # current < previous
                price_movmnt.append(0) # price down

            else: 
                price_movmnt.append(1) # price up

        self.stock['price_movmnt'] = price_movmnt
        return self.stock   
    
    
    # Beta values
    def beta_values(self, stock):        
        self.stock = stock        
        dr_stock = self.stock['dr_stock']
        dr_NSE = self.stock['dr_nse']
        # To handle 'ZeroDivisionError'
        dr_NSE = dr_NSE.replace(to_replace = {0:np.nan})
        self.stock['beta'] = round((dr_stock / dr_NSE), 2)
        # Filling all the NaNs by 0
        self.stock['beta'].fillna(0, inplace=True)
        return self.stock    
    

    # time-series prediction for the stock using prophet    
    def TS_close_price(self, stock):
        
        self.stock = stock

        self.stock['ds'] = pd.to_datetime(self.stock['Date'], format='%Y-%m-%d', errors='coerce')
        print('\n')
        print(self.stock['ds'].head())

        data = self.stock[['ds', 'Close']]
        data = data.rename(columns={'Close':'y_orig'})
        print('\n')
        print(data.head())    

        """log transform the ‘y’ variable('Close' price) to convert non-stationary data to stationary. 
        This also converts trends to more linear trends """

        #data['y'] = data['y_orig']

        data['y'] = np.log(data['y_orig'])
        print('\n')
        print(data.head())

        m = Prophet()
        m.fit(data)

        future = m.make_future_dataframe(periods=0, freq='d') # periods=0 as we don't need future predictions
        forecast = m.predict(future)
        print('\n')
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

        m.plot(forecast)
        m.plot_components(forecast)

        data[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
        print('\n')
        print(data.head()) 

        #stock['TS_close_pr'] = data['yhat']

        # Converting log-transformed 'yhat' to actual prices    
        self.stock['TS_close_pr'] = np.exp(data['yhat'])

        return self.stock, data

