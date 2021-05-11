#******************************************Rolling Funcs ******************************************
import numpy as np
def exp_2_avg(x): 
    x = x[::-1]
    weights = np.array([2**(len(x) - 1 - y) /(2**(len(x))-1) for y in range(len(x))])
    average = np.dot(np.array(x),weights)
    return average

def exp_e_avg(x): 
    x = x[::-1]
    weights = np.array([np.e**(len(x) - 1 - y) /(np.e**(len(x))-1) for y in range(len(x))])
    average = np.dot(np.array(x),weights)
    return average

def exp_x_avg(x,constant=10):
    x = x[::-1]
    weights = np.array([constant**(len(x) - 1 - y) /(constant**(len(x))-1) for y in range(len(x))])
    average = np.dot(np.array(x),weights)
    return average

def pond_avg(x): 
    x = x[::-1]
    weights = np.array([((len(x)-y))/(int(((len(x))*((len(x)+1))))/2) for y in range(len(x))])
    average = np.dot(np.array(x),weights)
    return average

def relu(x):
    return max(0,x)

def relu_neg(x):
    return min(0,x)

def RSI(x, agg=exp_e_avg):
    "vender 70-100, comprar 0-30, 30-70 neutral. x es el precio Adj close"
    x = np.array(x)
    today = x[1:]
    yesterday = x[:-1]
    up = today - yesterday
    down = up*(-1)
    up = up.clip(0,np.inf)
    down = down.clip(0,np.inf)
    rsi =  ( agg(up)/ (agg(up)+agg(down)) ) * 100 - 100/1+(agg)
    return rsi

def RS(x, agg=exp_e_avg):
    x = np.array(x)
    today = x[1:]
    yesterday = x[:-1]
    up = today - yesterday
    down = up*(-1)
    up = up.clip(0,np.inf)
    down = down.clip(0,np.inf)
    rs = agg(up)/agg(down)
    return rs

def K_line(x):
    "Stochastic Momentum Oscillator formula, sell above 80, buy under 20, se clcula sobre Adj close"

    closing = x[-1]
    l_n = min(x)
    h_n = max(x)
    k = (closing-l_n)/(h_n-l_n)
    return k 

def D_line(x):
    "otra es la media movil(3periodos) k_line"
    l_n = min(x)
    h_n = max(x)    
    return h_n/l_n

def CCI(x):
    """CCI = (Typical Price  -  20 Period SMA of TP) / (.015 x Mean Deviation)
    Typical Price (TP) = (High + Low + Close)/3
    Constant = .015
    cci > 100 vennder
    cci < -100 comprar
    -100<cci<100 neutro. se calcula sobre TP"""
    tp = x[-1]
    average = np.mean(x)
    const = 0.015
    md = abs(tp-average)/len(x)
    cci = (tp-average)/(const*md)
    return cci

def awesome_oscilator(x, short_period=5):
    "comprar ao > 0, vender ao < 0. se calcula sobre TP"
    ao = np.mean(x)-np.mean(x[-5:])
    return ao

def momentum(x):
    "comprar mom > 0, vender mom < 0"
    v = x[-1]
    vx = x[0]
    mom = v-vx
    return mom

def MACD(x):
    """MACD Line: (12-day EMA - 26-day EMA) 
        Signal Line: 9-day EMA of MACD Line
        MACD Histogram: MACD Line - Signal Line
        comprar macd > 0, vender macd < 0"""
    ema_12 = exp_e_avg(x[-12:])
    ema_26 = exp_e_avg(x)
    macd = ema_12-ema_26
    return macd

def MACD_x(x, short_period=12):
    """MACD Line: (12-day EMA - 26-day EMA) 
        Signal Line: 9-day EMA of MACD Line
        MACD Histogram: MACD Line - Signal Line"""
    ema_12 = exp_e_avg(x[-short_period:])
    ema_26 = exp_e_avg(x)
    macd = ema_12-ema_26
    return macd

def williams(x):
    """A reading above -20 is overbought (sell).
    A reading below -80 is oversold. (buy)
    An overbought or oversold reading doesn't mean the price will reverse. Overbought simply means the price is near the highs of its recent range, and oversold means the price is in the lower end of its recent range.
    Can be used to generate trade signals when the price and the indicator move out of overbought or oversold territory.
    %R = (Highest High - CurrentClose) / (Highest High - Lowest Low) x -100

    Highest High = Highest High for the user defined look-back period.
    Lowest Low = Lowest Low for the user defined look-back period."""
    closing = x[-1]
    l_n = min(x)
    h_n = max(x)
    R = (h_n - closing)/(h_n-l_n)
    return R

def ultimate(close,low,high):
    
    bp = close - min(low,)    

    R = (h_n - closing)/(h_n-l_n)


# ----------------------------------------- Performance Statistics -----------------------------------------
# also useful to portfolio management

def mdd(x):
    "Maximum drawdown, for absolute price, not the returns. closer to 0 is better"
    x = np.array(x)
    high_index = x.argmax()
    high = x.max()
    low = x[high_index:].min()
    result = abs((low-high)/high)
    return result

def RoMad(x):
    "Returns over maximum drawdown,(calmar) for absolute price, not the returns. the higher the better"
    ret = x.pct_change().mean()
    mad = mdd(x)
    result = ret/mad
    return result
#++++++++++++++++++++++++++++++++++++++++Functions to the Returns (pct_change)++++++++++++++++++++++++++++++++++++++++++
def sharpe_ratio(x):
    "sharpe, for a single stock, for the returns"
    ret = x.mean()
    std = x.std()
    result = ret/std
    return result

def sortino_ratio(x):
    "second order Lower  Paartial Moment"
    ret = x.mean()
    std_bad = x.clip(-np.inf,0).std()
    result = ret/std_bad
    return result

def omega_ratio(x):
    "ohm(r) = C(r)/P(r), call/put, 1+(mu-r)/E[max(r-x,0)]"
    "First order Lower  Paartial Moment"
    ret = x.mean()
    esp_bad = x.clip(-np.inf,0).abs().mean()
    result = 1 + ret/esp_bad
    return result

def tail_ratio(x):
    rets = x.sort_values(ignore_index=True)
    lenght = int(np.ceil((0.10*len(x))))
    right_tail = rets.tail(lenght)
    left_tail = rets.head(lenght)
    result = right_tail/left_tail
    return result

# VaR = -St(r+sigma*phi_1-alpha)
#St = precio de la accion en t
#r = rendimiento promedio diario
#sigma = volatilidad diaria
#phi_1-alpha = cuantil 1-alpha  de la normal norm.ppf(1-alpha, 0,1)

# r_p = (w'*mu_r), sigma²_p=w'Vw