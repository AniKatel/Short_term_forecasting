from pandas import DataFrame
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product


# ACF, PACF
def acf_pacf(data):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(data, lags=50, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(data, lags=50, ax=ax2)
    plt.show()


# тест Дики-Фуллера на наличие единичных корней
def adfuller(data):
    test = sm.tsa.adfuller(data)
    print('Тест Дики-Фуллера:')
    print('Значение статистики adf: ', test[0])
    print('p-значение: ', test[1])
    print('Критические значения: ', test[4])
    if test[0] > test[4]['5%']:
        print('Eсть единичные корни, ряд не стационарен')
    else:
        print('Eдиничных корней нет, ряд стационарен')


# Q-тест Льюинга-Бокса на белый шум
def ljungbox(data):
    q_test = sm.tsa.stattools.acf(data, qstat=True)
    print('Тест Льюнга-Бокса:')
    print(DataFrame({'Q-статистика': q_test[1], 'p-значение': q_test[2]}))


# поиск оптимальных параметров для SARIMA
def sarima_best_params(data, season_len, d, D, param_range, s_param_range):
    ps = range(0, param_range)
    qs = range(0, param_range)
    Ps = range(0, s_param_range)
    Qs = range(0, s_param_range)
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)

    results = []
    best_aic = float("inf")

    for param in tqdm(parameters_list):
        try:
            model = sm.tsa.SARIMAX(data, order=(param[0], d, param[1]),
                                   seasonal_order=(param[2], D, param[3], season_len)).fit(disp=False)
        except:
            print('Ошибка! Неправильные параметры:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    result_table = DataFrame(results)
    result_table.columns = ['параметры', 'aic']
    print('Наилучшие параметры:')
    print(result_table.sort_values(by='aic', ascending=True).head())
    return best_param
