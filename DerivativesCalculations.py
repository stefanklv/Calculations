# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:47:39 2022

@author: stefa
"""
import math 
import numpy as np
from scipy.stats import norm
import random

# Continuously compounded interest and dividend yield
def forward_cc(spot, interest, div, ttm):
    forward_price = spot * math.exp((interest - div)*ttm)
    return forward_price
    
# Discrete dividends and annually compounded interest
def forward_ann(spot, interest, div, ttm): 
    forward_price = spot * (1 + interest) ** ttm - div
    return forward_price
    
# Black Scholes
def d1(spot, exercise, interest, div, sigma, ttm): 
    d_1 = (np.log(spot / exercise) + (interest - div + 0.5 * sigma ** 2) * ttm) / (sigma * math.sqrt(ttm))
    return d_1 

def black_scholes_call(spot, exercise, interest, div, sigma, ttm): 
    d_1 = d1(spot=spot, exercise = exercise, interest=interest, div=div, sigma = sigma, ttm = ttm)
    d_2 = d_1 - sigma * math.sqrt(ttm) 
    normal_d1 = norm.cdf(d_1, loc=0, scale = 1) 
    normal_d2 = norm.cdf(d_2, loc=0, scale = 1) 
    call_price = spot * math.exp(-div * ttm) * normal_d1 - exercise * math.exp(-interest * ttm) * normal_d2
    return call_price

def black_scholes_put(spot, exercise, interest, div, sigma, ttm): 
    d_1 = d1(spot=spot, exercise = exercise, interest=interest, div=div, sigma = sigma, ttm = ttm)
    d_2 = d_1 - sigma * math.sqrt(ttm) 
    normal_minus_d1 = norm.cdf(-d_1, loc=0, scale = 1) 
    normal_minus_d2 = norm.cdf(-d_2, loc=0, scale = 1) 
    put_price = exercise * math.exp(-interest * ttm) * normal_minus_d2 - spot * math.exp(-div * ttm) * normal_minus_d1 
    return put_price

# Monte Carlo simulation for Pricing Asian Options

def asian_call(spot, exercise, interest, div, sigma, ttm, n_simulations, n_observations): 
    time_observations = [i / n_observations for i in range(0,n_observations+1)]
    
    discounted_payoffs = []
    
    for i in range(n_simulations): 
        random_numbers = [random.random() for i in range(1, n_observations+1)] # Creating list of random numbers
        random_inverse_norm = [norm.ppf(random_numbers[i]) for i in range(n_observations)] # Creating list of inverse normal numbers based on the random numbers
        
        # Creating list of stock prices for each observation
        stock_prices_t = []
        for i in range(0, n_observations): 
            if i == 0: 
                next_stock_price = 100 * math.exp((interest-div-sigma ** 2 / 2) * (time_observations[i+1]-time_observations[i]) + sigma*math.sqrt(time_observations[i+1]-time_observations[i])*random_inverse_norm[i])
            else: 
                next_stock_price = stock_prices_t[i-1] * math.exp((interest-div-sigma ** 2 / 2) * (time_observations[i+1]-time_observations[i]) + sigma*math.sqrt(time_observations[i+1]-time_observations[i])*random_inverse_norm[i])
           
            stock_prices_t.append(next_stock_price)     
               
        avg_stock_price = sum(stock_prices_t) / len(stock_prices_t)
        
        if avg_stock_price > exercise: 
            discounted_payoff = (avg_stock_price - exercise) * math.exp(-interest * ttm)
        else: 
            discounted_payoff = 0
        
        discounted_payoffs.append(discounted_payoff)
        
    avg_payoff = sum(discounted_payoffs) / len(discounted_payoffs)
        
    return avg_payoff