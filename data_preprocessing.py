import numpy as np
import pandas as pd

def extract_numeric_from_cols(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Функция, преобразующая данные, представленные в строковом формате, в числа
    '''
    df = df.copy()
    # Функция, которую мы будем применять к столбцам датафрейма
    def extract_numeric(x):    
        if type(x) == str:
            x = ''.join(filter(lambda y: str.isdigit(y) or y == '.', x.split()[0]))
            if x != '': 
                return float(x)
            else:
                return np.nan    
        elif type(x) == float:
            return x
        else:
            return np.nan    
   
    # Применяем функцию extract_numeric к столбцам, которые имеют тип данных str
    for column in ['mileage', 'engine', 'max_power', 'seats']:
        df[column] = df[column].apply(extract_numeric)
    return df  