from sklearn.metrics import mean_squared_error
import numpy as np


# метрика от ООО "ННФормат"
def nnfmetrics(pred, fact, plan):
    err_plan = np.abs(fact-plan).sum()
    err_pred = np.abs(fact-pred).sum()
    return 100*(err_plan-err_pred)/err_plan


# метрика MAPE
def mape(fact, pred):
    fact, pred = np.array(fact), np.array(pred)
    return np.mean(np.abs(fact - pred) / fact) * 100


# точность прогноза
def accuracy(fact, pred):
    return 100 - mape(fact, pred)


# метрика MSE
def mse(fact, pred):
    return mean_squared_error(fact, pred)