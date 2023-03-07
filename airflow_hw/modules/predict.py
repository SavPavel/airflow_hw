import os

import dill
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    z = os.listdir(f'{path}/data/models/')
    files = os.listdir(f'{path}/data/test/')

    with open(f'{path}/data/models/{z[0]}', 'rb') as file:
        model = dill.load(file)
    data = []
    for x in files:
        with open(f'{path}/data/test/{x}') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            rit = {'id': form['id'], 'pred': y[0]}
            data.append(rit)
    pred = pd.DataFrame(data)
    pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
