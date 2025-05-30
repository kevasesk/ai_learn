import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# Устанавливаем seed для воспроизводимости
np.random.seed(0)

# Генерируем данные
n_samples = 100
x1 = np.random.rand(n_samples)  # Первый признак
x2 = x1 + 0.01 * np.random.randn(n_samples)  # Второй признак, сильно коррелирующий с x1
y = 2 * x1 + 3 * x2 + 0.1 * np.random.randn(n_samples)  # Целевая переменная

# Создаем DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
X = data[['x1', 'x2']]

# Обучаем обычную линейную регрессию
model_lr = LinearRegression()
model_lr.fit(X, y)
print(f'Коэффициенты линейной регрессии: {model_lr.coef_}')

# Добавляем шум к y
y_noisy = y + 0.01 * np.random.randn(n_samples)

# Обучаем линейную регрессию на данных с шумом
model_lr_noisy = LinearRegression()
model_lr_noisy.fit(X, y_noisy)
print(f'Коэффициенты линейной регрессии с шумом: {model_lr_noisy.coef_}')

# Обучаем гребневую регрессию
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X, y)
print(f'Коэффициенты Ridge: {model_ridge.coef_}')

# Обучаем гребневую регрессию на данных с шумом
model_ridge_noisy = Ridge(alpha=1.0)
model_ridge_noisy.fit(X, y_noisy)
print(f'Коэффициенты Ridge с шумом: {model_ridge_noisy.coef_}')

# Сравнение коэффициентов
coefficients = pd.DataFrame({
    'Linear Regression': model_lr.coef_,
    'Linear Regression (noisy)': model_lr_noisy.coef_,
    'Ridge': model_ridge.coef_,
    'Ridge (noisy)': model_ridge_noisy.coef_
}, index=['x1', 'x2'])

print("\nСравнение коэффициентов:")
print(coefficients)

# Визуализация
coefficients.plot(kind='bar')
plt.title('Сравнение коэффициентов')
plt.ylabel('Значение коэффициента')
plt.show()