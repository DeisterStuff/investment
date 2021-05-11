import numpy as np
from functools import partial

def get_weights(n_assets,n_portfolios, sell=False):
    weights = np.random.random(size=(n_assets,n_portfolios))
    weights /= sum(weights)
    weights = np.stack(weights,axis=1)
    if sell:
        weights *= np.random.choice([-1, 1], size=weights.shape)
    return weights

def simulate_portfolios(weights, returns, risk_free=0, returns_periods=180, interval='d', sharpe=True, sortino=True, omega=False, tail=False):
    dic = dict()

    means = returns.mean() * returns_periods
    returns_porfolio = weights @ means + 1
    dic['returns'] = returns_porfolio
    if sharpe:
        volatility = (weights @ returns.T).std(1)
        dic['volatility'] = volatility
        dic['sharpe'] = (returns_porfolio-risk_free)/volatility
    if omega:
        returns_down = np.clip(returns, -np.inf,0)
        #returns_up = np.clip(returns, 0,np.inf)
        down_means = returns_down.abs().mean()*returns_periods
        #up_means = returns_up.abs().mean()*returns_periods
        rets_down = (weights @ down_means + 1 )
        # rets_up = (weights @ up_means + 1 )
        dic['returns down'] = rets_down
#         dic['omega'] = rets_up/rets_down
        dic['omega'] = 1 + (returns_porfolio-risk_free)/rets_down
    if sortino:
        returns_down = np.clip(returns, -np.inf,0)
        volatility_down = (weights @ returns_down.T).std(1)
        dic['volatility down'] = volatility_down
        dic['sortino'] = (returns_porfolio-risk_free)/volatility_down
    # if tail:
    #     rets = sorted(np.array(returns_porfolio))
    #     lenght = int(np.ceil((0.10*len(x))))
    #     right_tail = sum(rets[-lenght:])
    #     left_tail = sum(rets[:lenght])
    #     dic['tail'] = right_tail/left_tail
    # mean_returns = monthly_returns.mean()
    # cov_matrix = monthly_returns.cov()
    # precision_matrix = pd.DataFrame(inv(cov_matrix), index=stocks,
    # columns=stocks)
    # kelly_wt = precision_matrix.dot(mean_returns).values
    return dic
         
def factible_weights(P,weights,assets):
    array = np.unique(((weights*P)/assets).round(),axis=0)
    
    f = partial(np.dot,assets)
    p = np.stack(list(map(f,array)))
    
    w = (array*assets)/p.reshape(-1,1)
    return w

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

def anualidad(R,i,n):
    """[summary]

    Args:
        R (float): Renta periodica (ejemplo: 1000 al mes)
        i (float): tasa efectiva (ejemplo: 5% anual -> i = 0.05/12)
        n (int): numero de periodos (ejemplo: 12 meses)

    Returns:
        float :  acumulado

        example:
         anualidad(1000,0.05/12,17*12)
            >>> 12278.855491615914
    """
    factor = (1+i)**n
    anu = (factor-1)/i
    total = R*anu
    return total

