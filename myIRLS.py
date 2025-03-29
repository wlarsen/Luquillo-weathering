#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:26:34 2020

@author: wlarsen
"""
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model,Parameter

import statsmodels.formula.api as smf 
import statsmodels.api as sm

import pandas as pd


def myIRLS(X,Y,ww,method):
    
    # first of all
    # for any weight of 0, set to something small like 0.000001
    # then weights are 1/weight. 
    # ww[ww==0]=0.00001
    # ww = 1/ww
    
    X = np.array(X)
    Y = np.array(Y)
    ww = np.array(ww)
    
    nanmask = np.isnan(X) | np.isnan(Y) | np.isnan(ww)
    
    X = X[~nanmask]
    Y = Y[~nanmask]
    ww = ww[~nanmask]
    
    
    
    def bisquare(x,mar):
        # mar = mean absolute residual
        if mar ==0 :
            w = 1 if x== 0 else 0
        else:
            w = (1-(x/(6*mar))**2)**2 # bisquare
            w[x>6*mar] = 0 # replace weight with 0 if x esceeds 6times mar
        return w
    
    def welsch(x,mar):
        if mar == 0: 
            w = 1 if x == 0 else 0
        else: 
            w = np.exp(-(x/(4.4255*mar))**2)
        return w
    
    def cauchy(x,mar):
        if mar == 0:
            w = 1 if x== 0 else 0
        else: 
            w = 1/(1+(x/(3.536*mar))**2)
        return w

    def sincos(x, amplitude1, amplitude2, offset):
        out = [np.cos(i*2*np.pi)*amplitude1 + np.sin(i*2*np.pi)*amplitude2 + offset for i in x]
        # out = [np.cos(i*guess_freq-phase)*amplitude1 + np.sin(i*guess_freq-phase)*amplitude2 for i in x]
        return out
    
    wt = np.repeat(1,len(Y))
    wwwt = ww*wt

    
    formula = "y ~ sinx + cosx"
    fitdf = pd.DataFrame({'sinx':np.sin(X*2*np.pi),'cosx':np.cos(X*2*np.pi),'y':Y})
    model = smf.wls(formula=formula, weights=wwwt,data=fitdf) 
    # model = smf.ols(formula=formula,data=fitdf) 
    result = model.fit() 
    rsquared = result.rsquared
    residuals = result.resid

    
    wt_chg = 999
    iteration = 0
    
    while ((np.nanmax(wt_chg)>0.01)):# | (iteration<10)):# & ~all([iteration>10,rsquared>0.999]):
        if iteration>49:
            break
        iteration += 1
        old_wt = wt
        
        
        abs_residuals = np.abs(residuals)
        
        abs_residuals[abs_residuals==0] = np.nan
        
        mar = np.nanmedian(abs_residuals)
        
        
        if method == 'bisquare':
            wt = bisquare(abs_residuals,mar)
        if method == 'welsch':
            wt = welsch(abs_residuals,mar)
        if method == 'cauchy':
            wt = cauchy(abs_residuals,mar)
            
        wwwt = ww*wt
       
        formula = "y ~ np.sin(x*2*np.pi) + np.cos(x*2*np.pi)"
        fitdf = pd.DataFrame({'x':X, 'y':Y})
        model = smf.wls(formula=formula, weights=wwwt,data=fitdf) 
        result = model.fit()        
        
        rsquared = result.rsquared
        residuals = result.resid

        wt_chg = np.abs(wt-old_wt)
        

    finalfit = result
    
    return finalfit, model




def sincos(x, amplitude1, amplitude2,offset):
    out = [np.cos(i*2*np.pi)*amplitude1 + np.sin(i*2*np.pi)*amplitude2+offset for i in x]
    return out

def sincosfit(t,data,ww,irls=True,t_plot=np.arange(0,2.5,0.01)):
    t = np.array(t)
    data = np.array(data)
    ww = np.array(ww)
    
    nanmask = np.isnan(t) | np.isnan(data) | np.isnan(ww)
    
    t = t[~nanmask] 
    data = data[~nanmask]
    ww = ww[~nanmask]
    

    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_offset = np.mean(data)
    p0=[guess_amplitude, guess_amplitude, guess_offset]
    if irls==True:
        fit,model = myIRLS(t,data,ww,'welsch')
        
        resultfunc = fit.predict()
        resultfunc_plot = fit.predict(exog=dict(x=t_plot))
    else:

         formula = "y ~ sinx + cosx"
         fitdf = pd.DataFrame({'sinx':np.sin(t*2*np.pi),'cosx':np.cos(t*2*np.pi),'y':data})
         
         formula = "y ~ np.sin(x*2*np.pi) + np.cos(x*2*np.pi)"
         fitdf = pd.DataFrame({'x':t, 'y':data})
         
         model = smf.ols(formula=formula,data=fitdf) 
         fit = model.fit()
         
         resultfunc = fit.predict()
         resultfunc_plot = fit.predict(exog=dict(x=t_plot))
 

    
    a = fit.params[2] # coeff for cos
    b = fit.params[1]# coeff for sin
        
    res_sumsq = np.sum((np.array(data)-resultfunc)**2)
    tot_sumsq = np.sum((data-np.mean(data))**2)
    rsquared = 1-(res_sumsq/tot_sumsq)
    
#    rmse = np.sqrt(res_sumsq/len(data))
    
    try:
        covmat = np.matrix(fit.normalized_cov_params)[1:,1:] # covariance matrix includes intercept, a1 and a2
        covmat = np.matrix(fit.cov_params())[1:,1:] # covariance matrix includes intercept, a1 and a2
        corr = covmat[1,0]

    except:
        print('fit.covar is empty or something')

    

    sterror_amps = np.sqrt(np.diag(covmat))


    amp = np.sqrt(a**2+b**2)
    
    #  assume no correlation between a and b
    sterror = np.sqrt((a/amp)**2 *sterror_amps[1]**2+(b/amp)**2 * sterror_amps[0]**2)#/np.sqrt(len(t))

    
    
    phaseshift = np.arctan(b/a)/(2*np.pi)
    
    statsdict = {'amp':amp,'phaseshift':phaseshift,'rsquared':rsquared,'sterror':sterror}
    
    return fit, statsdict, resultfunc_plot



