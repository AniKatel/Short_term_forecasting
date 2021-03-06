from pandas import read_csv
from metrics import nnfmetrics, mape, accuracy, mse
from forecasting_plot import plot_results
import datetime
import statsmodels.api as sm


# аддитивный метод Хольта-Уинтерса
def additiveHoltWinters(params, ts, p, h):
    alpha, beta, gamma = params
    # инициализация
    # уровень ряда
    level = sum(ts[0:p]) / p
    # тренд
    trend = (sum(ts[p:2 * p]) - sum(ts[0:p])) / p ** 2
    # сезонность
    season = [ts[i] - level for i in range(p)]
    # предсказанные значения
    predict = []
    # промежуточные значения
    fitted_values = []
    # уточнение коэффициентов
    for i in range(0, len(ts) - p):
        season_value = season.pop(0)
        fitted_values.append(level + trend + season_value)
        new_level = alpha * (ts[i + p] - season_value) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season.append(gamma * (ts[i + p] - new_level) + (1 - gamma) * season_value)
        level = new_level
    resid = ts[p:] - fitted_values
    # рассматриваются остатки за последние два месяца
    resid = resid[-168 * 4 * 2:]
    # построение AR(1) модели для остатков
    model = sm.tsa.ARIMA(resid, order=(1, 0, 0)).fit(disp=False)
    # прогнозирование остатков
    res_pred = model.predict(len(resid), len(resid) + h - 1)
    # построение прогноза
    for i in range(0, h):
        predict.append(res_pred[i] + level + (i + 1) * trend + season[i % p])
    return predict


# мультипликативный метод Хольта-Уинтерса
def multiplicativeHoltWinters(params, ts, p, h):
    alpha, beta, gamma = params
    # инициализация
    # уровень ряда
    level = sum(ts[0:p]) / p
    # тренд
    trend = (sum(ts[p:2 * p]) - sum(ts[0:p])) / p ** 2
    # сезонность
    season = [ts[i] / level for i in range(p)]
    # предсказанные значения
    predict = []
    # промежуточные значения
    fitted_values = []
    # уточнение коэффициентов
    for i in range(0, len(ts) - p):
        season_value = season.pop(0)
        fitted_values.append((level + trend) * season_value)
        new_level = alpha * (ts[i + p] / season_value) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season.append(gamma * (ts[i + p] / new_level) + (1 - gamma) * season_value)
        level = new_level
    resid = ts[p:] - fitted_values
    # рассматриваются остатки за последние два месяца
    resid = resid[-168 * 4 * 2:]
    # построение AR(1) модели для остатков
    model = sm.tsa.ARIMA(resid, order=(1, 0, 0)).fit(disp=False)
    # прогнозирование остатков
    res_pred = model.predict(len(resid), len(resid) + h - 1)
    for i in range(0, h):
        predict.append(res_pred[i] + (level + (i + 1) * trend) * season[i % p])
    return predict


# временной ряд с датасетом data1.csv
dataset = read_csv('../data1.csv', index_col=0, parse_dates=[0], dayfirst=True)
# фактическое потребление электроэнергии - временной ряд для прогноза
fact = dataset.fact
# план потребления электроэнергии
plan = dataset.plan
# длина периода сезонной составляющей
p = 168
# тренировочный датасет train_set, данные с 14.01.2019 по 25.11.2019
train_start_date = datetime.datetime(2019,1,14,0,0,0)
train_end_date = datetime.datetime(2019,11,25,23,0,0)
train_set = fact[train_start_date:train_end_date]
# тестовый датасет test_set, данные за прогнозный день 27.11.2019
pred_date = datetime.date(2019,11,27)
pred_end_date = train_end_date + datetime.timedelta(days=2)
pred_start_date = pred_end_date - datetime.timedelta(hours=23)
test_set = fact[pred_start_date:pred_end_date]
# план предприятия plan_set для сверки, данные за прогнозный день 27.11.2019
plan_set = plan[pred_start_date:pred_end_date]
# выбор метода
method = additiveHoltWinters
# найденные постоянные сглаживания
best_params = [[0.06846567, 0.00032291, 0.26071966]]
# прогнозирование
pred = method(best_params[0], train_set, p, 48)[24:]
# оценка прогноза по метрикам
nnf_val = nnfmetrics(pred, test_set, plan_set)
mse_val = mse(test_set, pred)
mape_val = mape(test_set, pred)
acc_val = accuracy(test_set, pred)
print('Оценка по NNFMETRICS = ', nnf_val)
print('Оценка по MSE = ', mse_val)
print('Оценка по MAPE = ', mape_val)
print('Точность прогноза = ', acc_val)
# отрисовка графика
plot_results(pred_date.strftime('%d-%m-%Y'), test_set, pred, plan_set)