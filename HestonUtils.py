import numpy as np
import torch
import QuantLib as ql
import time

import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint

def QuantlibHestonPrice(spot, strike, current_date, maturity_date,
                v_0, kappa, v_bar, sigma, rho,
                rf_rate=0, dv_rate=0,
                option_type = ql.Option.Call, day_count = ql.Actual365Fixed(), calendar = ql.UnitedStates()):
    ql.Settings.instance().evaluationDate = current_date
    
    payoff= ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rf_handle = ql.YieldTermStructureHandle(ql.FlatForward(current_date, rf_rate, day_count))
    dv_handle = ql.YieldTermStructureHandle(ql.FlatForward(current_date, dv_rate, day_count))
    heston_process = ql.HestonProcess(rf_handle, dv_handle, spot_handle,
                                      v_0, kappa, v_bar, sigma, rho)
    
    pricing_engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    european_option.setPricingEngine(pricing_engine)
    heston_price = european_option.NPV()
    
    return heston_price


def convertNNtoQLparams(nn_parameters):
    nn_parameters = nn_parameters.reshape(-1,9)
    M = len(nn_parameters)
    current_dates = np.array([ql.Date(1,1,2019)]*M).reshape(-1,1)
    maturity_dates = current_dates + np.round(nn_parameters[:,1] * 365).astype(int).reshape(-1,1)
    Ss = nn_parameters[:,0].reshape(-1,1)
    Ks = np.ones(M).reshape(-1,1)
    ql_parameters = np.hstack((Ss, Ks, current_dates, maturity_dates, nn_parameters[:,2:]))
    return ql_parameters

def Feller(x):
    return 2*x[1]*x[2] - x[3]**2

feller_con = NonlinearConstraint(Feller, lb=0, ub=np.inf)