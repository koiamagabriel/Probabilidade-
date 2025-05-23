import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, gaussian_kde


csv_path = 'Downloads/registro01.csv'


df = pd.read_csv(csv_path)
x = df['x0']
y = df['x9']


# a) Estatísticas descritivas

stats = pd.DataFrame({
    'Estatística': ['Média', 'Variância (pop.)', 'Desvio Padrão (pop.)', 'Mediana'],
    'x0': [x.mean(), x.var(ddof=0), x.std(ddof=0), x.median()],
    'x9': [y.mean(), y.var(ddof=0), y.std(ddof=0), y.median()]
})
print("\n(a) Estatísticas descritivas para x0 e x9:")
print(stats.to_string(index=False))


# b) Histogramas

plt.figure()
plt.hist(x, bins='auto', edgecolor='black')
plt.title('Histograma de x0')
plt.xlabel('x0')
plt.ylabel('Frequência')
plt.show()

plt.figure()
plt.hist(y, bins='auto', edgecolor='black')
plt.title('Histograma de x9')
plt.xlabel('x9')
plt.ylabel('Frequência')
plt.show()


# c) Boxplots 

plt.figure()
plt.boxplot([x, y], tick_labels=['x0', 'x9'])
plt.title('Boxplots de x0 e x9')
plt.ylabel('Valores')
plt.show()


# d) Coeficiente de Correlação

corr = x.corr(y)
print(f"\n(d) Coeficiente de correlação de Pearson entre x0 e x9: {corr:.6f}")


# e) Testes de Normalidade


stat_x_sw, p_x_sw = shapiro(x)
stat_y_sw, p_y_sw = shapiro(y)

stat_x_dp, p_x_dp = normaltest(x)
stat_y_dp, p_y_dp = normaltest(y)

norm_results = pd.DataFrame({
    'Teste': ['Shapiro-Wilk', 'Shapiro-Wilk', "D’Agostino-Pearson", "D’Agostino-Pearson"],
    'Variável': ['x0', 'x9', 'x0', 'x9'],
    'Estatística': [stat_x_sw, stat_y_sw, stat_x_dp, stat_y_dp],
    'p-valor': [p_x_sw, p_y_sw, p_x_dp, p_y_dp]
})
print("\n(e) Testes de normalidade para x0 e x9:")
print(norm_results.to_string(index=False))


# f) Histograma + Densidade KDE

plt.figure()
plt.hist(x, bins='auto', density=True, alpha=0.6, edgecolor='black')
kde_x = gaussian_kde(x)
grid_x = np.linspace(x.min(), x.max(), 1000)
plt.plot(grid_x, kde_x(grid_x), lw=2)
plt.title('Histograma e densidade de x0')
plt.xlabel('x0')
plt.ylabel('Densidade')
plt.show()

plt.figure()
plt.hist(y, bins='auto', density=True, alpha=0.6, edgecolor='black')
kde_y = gaussian_kde(y)
grid_y = np.linspace(y.min(), y.max(), 1000)
plt.plot(grid_y, kde_y(grid_y), lw=2)
plt.title('Histograma e densidade de x9')
plt.xlabel('x9')
plt.ylabel('Densidade')
plt.show()
