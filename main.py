import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from fastapi import FastAPI
import uvicorn



def data(ticker):
    
    df = pdr.DataReader(ticker, data_source='yahoo', start='2016-01-01')
    df.index = pd.to_datetime(df.index, format="%Y/%m/%d")
    df = pd.Series(df['Close'])
    last_day=df[-1]
    return df , last_day

def best_order(df):
    
    model = pm.auto_arima(df, start_p=0, start_q=0, test='adf', max_p=2, max_q=2, m=1,d=None,seasonal=False   
                      ,start_P=0,D=0, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
    order = model.order
    return order

def model(df,order,last_day):
    model = ARIMA(df, order=order)
    model_fit = model.fit(disp=0)
    fc ,se, conf = model_fit.forecast(1)
    diff = fc - last_day
    return fc , diff

def overall(ticker):
    df,last_day = data(ticker)
    order  = best_order(df)
    fc , diff =model(df , order ,last_day)
    return last_day , fc , diff 





app = FastAPI()

@app.get('/')

def index():
    return {'message': 'Hello!'}

@app.post('/predict')

async def predict_price(ticker:str):
    
    last_day , fc , diff = overall(ticker)
    return {'prediction':fc[0],'difference:':diff[0]}
       
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    