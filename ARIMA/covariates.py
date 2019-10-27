import pandas as pd
import numpy as np
from pandas import datetime
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime(x, '%b %d, %Y')

data = pd.read_csv("SP500.csv", index_col=0, usecols=["Date", "Price"], parse_dates=[0], date_parser=parser)
data['Price'] = data['Price'].str.replace(',','').astype(float)
data = data.reindex(index=data.index[::-1])

data_2 = pd.read_csv("USD_CNY.csv", index_col=0, usecols=["Date", "Price"], parse_dates=[0], date_parser=parser)
data_2 = data_2.reindex(index=data_2.index[::-1])

dataset = data.values
size = int(len(dataset) * 0.90)
train, test = dataset[0:size], dataset[size:len(dataset)]
history = [x for x in train]
predictions = list()

exog_dataset = (data_2.values)[0:size] # trim to match dataset size
exog = [x for x in exog_dataset]

for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0), exog=exog)
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	# print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
# pyplot.plot(dataset, color='blue')
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
