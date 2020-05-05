import matplotlib.pyplot as plt
import numpy as np


# график фактических значений, прогнозных и плана предприятия
def plot_results(date, fact, pred, plan):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    x = list(i for i in range(24))
    y1 = np.array(fact)
    y2 = np.array(pred)
    y3 = np.array(plan)
    ax.plot(x, y1, label='фактические значения')
    ax.plot(x, y2, label='прогнозные значения')
    ax.plot(x, y3, label='план предприятия')
    plt.title('Построение прогноза на ' + date, fontsize=17)
    plt.xlabel('время суток', fontsize=15, color='blue')
    plt.ylabel('объемы электропотребления', fontsize=15, color='blue')
    plt.legend()
    plt.show()