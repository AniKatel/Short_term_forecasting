from pandas import read_csv
from metrics import nnfmetrics, mape, accuracy, mse
from forecasting_plot import plot_results
import datetime
import statsmodels.api as sm


# центрированное скользящее среднее вокруг ts[j], где ts - временной ряд с периодом p
def Centered_moving_average(ts, p, j):
    half_p = int(p / 2)
    return (1 / p) * sum(ts[i] for i in range(j - half_p, j + half_p))


# аддитивный метод Хольта-Уинтерса с двойной сезонностью
def double_seasonal_additiveHoltWinters(params, ts, p1, p2, h):
    alpha, beta, gamma, delta = params
    # инициализация
    # уровень ряда
    level = sum(ts[0:2 * p2]) / (2 * p2)
    # тренд
    trend = (sum(ts[p2:2 * p2]) - sum(ts[0:p2])) / p2 ** 2
    # сезонность 1
    season1 = [0] * p1
    for i in range(p1):
        season1[i] = sum(ts[i + (1 + j) * p1] - Centered_moving_average(ts, p1, i + (1 + j) * p1) for j in range(7)) / 7
    # сезонность 2
    season2 = [0] * p2
    for i in range(p2):
        season2[i] = sum(ts[i + (1 + j) * p2] - Centered_moving_average(ts, p2, i + (1 + j) * p2) - season1[i % p1]
                         for j in range(2)) / 2
    # предсказанные значения
    predict = []
    # промежуточные результаты
    fitted_values = []
    # уточнение коэффициентов
    for i in range(0, len(ts) - p2):
        season1_value = season1.pop(0)
        season2_value = season2.pop(0)
        fitted_values.append(level + trend + season1_value + season2_value)
        new_level = alpha * (ts[i + p2] - season1_value - season2_value) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season1.append(gamma * (ts[i + p2] - new_level - season2_value) + (1 - gamma) * season1_value)
        season2.append(delta * (ts[i + p2] - new_level - season1_value) + (1 - delta) * season2_value)
        level = new_level
    resid = ts[p2:] - fitted_values
    # рассматриваются остатки за последние два месяца
    resid = resid[-168 * 4 * 2:]
    # построение AR(1) модели для остатков
    model = sm.tsa.ARIMA(resid, order=(1, 0, 0)).fit(disp=False)
    # прогнозирование остатков
    res_pred = model.predict(len(resid), len(resid) + h - 1)
    # построение прогноза
    for i in range(0, h):
        predict.append(res_pred[i] + level + (i + 1) * trend + season1[i % p1] + season2[i % p2])
    return predict


# мультипликативный метод Хольта-Уинтерса с двойной сезонностью
def double_seasonal_multiplicativeHoltWinters(params, ts, p1, p2, h):
    alpha, beta, gamma, delta = params
    # инициализация
    # уровень ряда
    level = sum(ts[0:2 * p2]) / (2 * p2)
    # тренд
    trend = (sum(ts[p2:2 * p2]) - sum(ts[0:p2])) / p2 ** 2
    # сезонность 1
    season1 = [0]*p1
    for i in range(p1):
        season1[i] = sum(ts[i + (1 + j) * p1] / Centered_moving_average(ts, p1, i + (1 + j) * p1) for j in range(7)) / 7
    # сезонность 2
    season2 = [0]*p2
    for i in range(p2):
        season2[i] = sum(ts[i + (1 + j) * p2] / (Centered_moving_average(ts, p2, i + (1 + j) * p2) * season1[i % p1])
                         for j in range(2)) / 2
    # предсказанные значения
    predict = []
    # промежуточные результаты
    fitted_values = []
    # уточнение коэффициентов
    for i in range(0, len(ts) - p2):
        season1_value = season1.pop(0)
        season2_value = season2.pop(0)
        fitted_values.append((level + trend) * season1_value * season2_value)
        new_level = alpha * (ts[i + p2] / (season1_value * season2_value)) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        season1.append(gamma * (ts[i + p2] / (new_level * season2_value)) + (1 - gamma) * season1_value)
        season2.append(delta * (ts[i + p2] / (new_level * season1_value)) + (1 - delta) * season2_value)
        level = new_level
    resid = ts[p2:] - fitted_values
    # рассматриваются остатки за последние два месяца
    resid = resid[-168 * 4 * 2:]
    # построение AR(1) модели для остатков
    model = sm.tsa.ARIMA(resid, order=(1, 0, 0)).fit(disp=False)
    # прогнозирование остатков
    res_pred = model.predict(len(resid), len(resid) + h - 1)
    # построение прогноза
    for i in range(0, h):
        predict.append(res_pred[i] + (level + (i + 1) * trend) * season1[i % p1] * season2[i % p2])
    return predict


# временной ряд с датасетом data1.csv
dataset = read_csv('../data1.csv', index_col=0, parse_dates=[0], dayfirst=True)
# фактическое потребление электроэнергии - временной ряд для прогноза
fact = dataset.fact
# план потребления электроэнергии
plan = dataset.plan
# длина периодов сезонных составляющих
p1 = 24
p2 = 168
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
method = double_seasonal_multiplicativeHoltWinters
# найденные постоянные сглаживания
best_params = [[0.334097458, 0.000291639411, 0.278284232, 0.123083456]]
# прогнозирование
pred = method(best_params[0], train_set, p1, p2, 48)[24:]
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