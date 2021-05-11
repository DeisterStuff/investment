import pandas as pd
from itertools import product
def create_shifts(work_df, columns, shifts):
    """
    work_df: un DataFrame
    column: str or iterable, columna o columnas a la cual se quiere applicar los shift
    shifts: iterable con los numeros de shifts a aplicar
    """
    shift_df = pd.concat(map(lambda sh: work_df[columns].shift(sh), shifts), axis=1)
    if isinstance(columns,str):
        shift_df.columns = [f"{columns}_shift_{sh}" for sh in shifts]
    if isinstance(columns,list):
            shift_df.columns = [f"{i}_shift_{sh}" for sh,i in list(product(shifts,columns))]
    return shift_df

# def create_rollings(work_df, columns, agg_funcs, windows, min_periods=None):
#     """
#     work_df: un DataFrame
#     column: str or arraylike, columna a la cual se quiere applicar los rollings
#     agg_func: str or list , funcion o lista de funciones de agregacion de los rollings
#     windows: iter, iterable con los numeros de ventanas para hacer los rollings
#     min_periods: int, periodos minimos para agregar el rolling
#     Return:
#     DataFrame  con los las columnas hechas rolling
#     """
#     if not isinstance(columns,list):
#         columns = [columns]
#     if not isinstance(agg_funcs,list):
#         agg_funcs = [agg_funcs]
#     rolling_df = pd.concat(map(lambda w: work_df[columns].rolling(w, min_periods).agg(agg_funcs), windows), axis=1)
#     rolling_df.columns = [f'rolling_{x}_{y}_{z}' for z,x,y in product(windows,columns,agg_funcs)]
#     return rolling_df

def create_rollings(work_df, columns, agg_funcs, windows, min_periods=None):
    """
    work_df: un DataFrame
    column: str or arraylike, columna a la cual se quiere applicar los rollings
    agg_func: str or list , funcion o lista de funciones de agregacion de los rollings
    windows: iter, iterable con los numeros de ventanas para hacer los rollings
    min_periods: int, periodos minimos para agregar el rolling
    Return:
    DataFrame  con los las columnas hechas rolling
    """

    if isinstance(columns,list) and len(columns)==1:
        columns = columns[0]
        
#     if not isinstance(agg_funcs,list):
#         agg_funcs = [agg_funcs]
    if ((not isinstance(agg_funcs,list)) and (not isinstance(agg_funcs,dict)) ):
        agg_funcs = [agg_funcs]
    agg_names = agg_funcs
    if isinstance(agg_funcs,dict):
        agg_names = agg_funcs.keys()
        
    rolling_df = pd.concat(map(lambda w: work_df[columns].rolling(w, min_periods).agg(agg_funcs), windows), axis=1)
    if not isinstance(columns,list):
        columns = [columns]
    rolling_df.columns = [f'rolling_{x}_{y}_{z}' for z,x,y in product(windows,columns,agg_funcs)]
    return rolling_df

def bollinger_bands(work_df, column, period=20):
    data = pd.DataFrame()
    data[f"close_{column}"] = work_df[column].copy()
    data.fillna(method='ffill', inplace=True)
    data[f'mean_{column}_{period}'] = data[f"close_{column}"].rolling(window=period).mean()
    data[f'std_{column}_{period}'] = data[f"close_{column}"].rolling(window=period).std() 

    data[f'upper_{column}'] = data[f'mean_{column}_{period}'] + (data[f'std_{column}_{period}'] * 2)
    data[f'lower_{column}'] = data[f'mean_{column}_{period}'] - (data[f'std_{column}_{period}'] * 2)
    return data 