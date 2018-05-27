# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:23:58 2017

@author: Aditya
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.optimize as spo
import numpy as np 

def f(X):
    Y = (X - 1.5)**2 + 0.5 
    
    print ('X  ' + str(X) + 'Y ' + str (Y))
    return Y

def test_run():
    guess = 2.0
    min_result = spo.minimize(f, guess, method = 'SLSQP', options = {'disp': True })
    print (str (min_result.x) + '' + str (min_result.fun))
    
    
    Xplot = np.linspace (0.5, 2.5 , 21)
    Yplot = f (Xplot)
    plt.plot (Xplot, Yplot)
    plt.plot(min_result.x , min_result.fun,'ro')
    plt.show ()
    
test_run()

