from pandas import read_csv
import statsmodels.api as sm
import datetime
from metrics import nnfmetrics, mape, accuracy, mse
from sarima.time_series_analysis import acf_pacf, adfuller, ljungbox, sarima_best_params
from forecasting_plot import plot_results


# временной ряд с датасетом data1.csv
dataset = read_csv('../data1.csv', index_col=0, parse_dates=[0], dayfirst=True)
# фактическое потребление электроэнергии - временной ряд для прогноза
fact = dataset.fact
# план потребления электроэнергии
plan = dataset.plan
# в тренировочный датасет входят данные с 01.01.2019 по 25.11.2019
train_start_date = datetime.datetime(2019,1,1,0,0,0)
train_end_date = datetime.datetime(2019,11,25,23,0,0)
# в тестовый датасет входят данные за прогнозный день 27.11.2019
pred_date = datetime.date(2019,11,27)
pred_end_date = train_end_date + datetime.timedelta(days=2)
pred_start_date = pred_end_date - datetime.timedelta(hours=23)
test_set = fact[pred_start_date:pred_end_date]
# план предприятия plan_set для сверки, данные за прогнозный день 27.11.2019
plan_set = plan[pred_start_date:pred_end_date]
# разбиение временного ряда fact на 24 датасета - для каждого часа
fact_by_hours = []
for i in range(0,24):
    fact_by_hours.append(fact[i::24])
# список для хранения прогнозных значений
total_pred = []
# найденные параметры для 27.11.2019:
best_params = [(1, 2, 2, 2), (6, 3, 1, 2), (3, 5, 1, 2), (2, 5, 1, 1), (0, 2, 1, 2), (2, 4, 1, 1),
               (0, 2, 2, 2), (5, 1, 1, 2), (3, 2, 1, 2), (5, 2, 1, 2), (1, 4, 1, 2), (3, 1, 1, 2),
               (3, 1, 1, 2), (3, 1, 1, 2), (0, 2, 1, 2), (1, 2, 2, 1), (1, 3, 1, 1), (0, 2, 1, 2),
               (0, 2, 2, 1), (0, 2, 1, 2), (5, 2, 2, 1), (1, 1, 1, 2), (3, 1, 2, 2), (4, 1, 1, 2)]
# построение SARIMA для каждого из 24 часов
for j in range(0,24):
    # определение тренировочного датасета для j-го часа
    train_set = fact_by_hours[j][train_start_date:train_end_date]
    # изучение свойств ряда
    # стационарность
    adfuller(train_set)
    # автокорреляции и частные автокорреляции
    acf_pacf(train_set)
    # ряд нестационарен, берется разность
    train_set_diff = train_set.diff(periods=1).dropna()
    # проверка свойств ряда из разностей
    adfuller(train_set_diff)
    acf_pacf(train_set_diff)
    # ряд стационарен, можно переходить к поиску оптимальных параметров
    # best_params.append(sarima_best_params(train_set, 1, 0, 7, 3))
    # построение модели
    model = sm.tsa.SARIMAX(train_set, order=(best_params[j][0], 1, best_params[j][1]),
                          seasonal_order=(best_params[j][2], 0, best_params[j][3], 7)).fit(disp=False)
    # вывод информации о модели
    print(model.summary())
    # проверка остатков модели на случайность
    ljungbox(model.resid)
    acf_pacf(model.resid)
    # SARIMA прогноз на конкретный час
    hour_pred = model.forecast(2)[-1]
    # добавление прогноза на час к итоговому прогнозу
    total_pred.append(hour_pred)
# оценка прогноза по метрикам
nnf = nnfmetrics(total_pred, test_set, plan_set)
mse = mse(test_set, total_pred)
mape = mape(test_set, total_pred)
acc = accuracy(test_set, total_pred)
print('Оценка по NNFMETRICS = ', nnf)
print('Оценка по MSE = ', mse)
print('Оценка по MAPE = ', mape)
print('Точность прогноза = ', acc)
# отрисовка графика
plot_results(pred_date.strftime('%d-%m-%Y'), test_set, total_pred, plan_set)
