import requests
import numpy as np
import pandas as pd

def get_fng_index():
    """
    """
    r = requests.get('https://api.alternative.me/fng/?limit=0')
    fng = r.json()['data'][::-1]
    df = pd.json_normalize(fng).iloc[:,:-1]
    df['timestamp'] = df['timestamp'].astype(int)
    df['value'] = df['value'].astype(int)
    df = df.replace(['Extreme Fear','Fear','Neutral','Greed','Extreme Greed'],[0, 1, 2, 3,4])
    return df
    
def split_data(
        data: np.ndarray,
        past_history: int,
        future_target: int,
        split_percent:int = 80
    ):
    """
    """
    input_data = []
    output_data = []
    future_target = 0 if future_target <= 1 else future_target - 1
    
    for i in range(past_history, len(data)+1): 
        indices = range(i-past_history, i)

        input_data.append(np.reshape(data[indices], (past_history, data.shape[1])))
        try:
            output_data.append(data[i+future_target][0])
        except:
            output_data.append(np.nan)

    input_data, output_data = np.array(input_data), np.array(output_data)

    split_rate = int(len(input_data) * (split_percent / 100))

    return (
        input_data[:split_rate],  # x_train
        input_data[split_rate:],  # x_test
        output_data[:split_rate], # y_train
        output_data[split_rate:]  # y_test
    )
