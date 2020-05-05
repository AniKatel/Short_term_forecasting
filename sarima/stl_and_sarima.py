from pandas import read_csv
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import rstl
import datetime
from metrics import nnfmetrics, mape, accuracy, mse
from sarima.time_series_analysis import acf_pacf, adfuller, ljungbox, sarima_best_params
from forecasting_plot import plot_results


# расчет коэффициентов линейного тренда
def linear_trend_params(data):
    y = data
    x = range(0, len(y))
    x = sm.add_constant(x)
    trend_model = sm.OLS(y, x).fit()
    return trend_model.params


# отрисовка графика STL-разложения
def plot_stl(ts, trend, trend_params, seasonal, remainder):
    fig = plt.figure(figsize=(12, 6))
    x = np.array(ts.index)
    y = np.array(ts)
    ax1 = fig.add_subplot(411)
    ax1.plot(x, y)
    ax2 = fig.add_subplot(412)
    y1 = np.array(trend)
    y2 = trend_params[0] + trend_params[1] * range(0, len(trend))
    ax2.plot(x, y1, x, y2)
    ax3 = fig.add_subplot(413)
    y = np.array(seasonal)
    ax3.plot(x,y)
    ax4 = fig.add_subplot(414)
    y = np.array(remainder)
    ax4.plot(x, y)
    plt.show()


# временной ряд с датасетом data1.csv
dataset = read_csv('../data1.csv', index_col=0, parse_dates=[0], dayfirst=True)
# фактическое потребление электроэнергии - временной ряд для прогноза
fact = dataset.fact
# план потребления электроэнергии
plan = dataset.plan
# тренировочный датасет train_set, данные за месяц с 25.10.2019 по 25.11.2019
train_start_date = datetime.datetime(2019,10,25,0,0,0)
train_end_date = datetime.datetime(2019,11,25,23,0,0)
train_set = fact[train_start_date:train_end_date]
# тестовый датасет test_set, данные за прогнозный день 27.11.2019
pred_date = datetime.date(2019,11,27)
pred_end_date = train_end_date + datetime.timedelta(days=2)
pred_start_date = pred_end_date - datetime.timedelta(hours=23)
test_set = fact[pred_start_date:pred_end_date]
# план предприятия plan_set для сверки, данные за прогнозный день 27.11.2019
plan_set = plan[pred_start_date:pred_end_date]
# STL-декомпозиция
freq = 168
stl_dec = rstl.STL(train_set, freq, "periodic")
trend = stl_dec.trend
seasonal = stl_dec.seasonal
remainder = stl_dec.remainder
# определение коэффициентов линейного тренда
trend_params = linear_trend_params(trend)
# прогноз трендовой составляющей
trend_pred = []
for j in range(24, 48):
    trend_pred.append(trend_params[0] + trend_params[1] * (len(train_set) + j))
# прогноз сезонной составляющей
seasonal_pred = seasonal[-144:-120]
# отрисовка графика STL
plot_stl(train_set, trend, trend_params, seasonal, remainder)
# прогноз случайной составляющей с помощью ARIMA
# отрисовка графика остатков
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()
x = remainder.index
y = remainder
ax.plot(x, y)
plt.show()
# изучение свойств остатков:
# проверка на стационарность
adfuller(remainder)
# автокорреляции и частные автокорреляции
acf_pacf(remainder)
# поиск лучших параметров
# sarima_best_params(remainder, 0, 0, 4, 3)
# найденные параметры для 27.11.2019:
best_params = [2, 3, 1, 2]
# построение модели
model = sm.tsa.SARIMAX(remainder, order=(best_params[0], 0, best_params[1]),
                       seasonal_order=(best_params[2], 0, best_params[3], 24)).fit(disp=False)
# вывод информации о модели
print(model.summary())
# проверка остатков модели на случайность
ljungbox(model.resid)
acf_pacf(model.resid)
# SARIMA прогноз остатков
remainder_pred = model.forecast(48)
# итоговый прогноз
total_pred = trend_pred + seasonal_pred + remainder_pred[24:]
# оценка прогноза по метрикам
nnf_val = nnfmetrics(total_pred, test_set, plan_set)
mse_val = mse(test_set, total_pred)
mape_val = mape(test_set, total_pred)
acc_val = accuracy(test_set, total_pred)
print('Оценка по NNFMETRICS = ', nnf_val)
print('Оценка по MSE = ', mse_val)
print('Оценка по MAPE = ', mape_val)
print('Точность прогноза = ', acc_val)
# отрисовка графика
plot_results(pred_date.strftime('%d-%m-%Y'), test_set, total_pred, plan_set)