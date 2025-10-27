import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("Advertising.csv")
print(data.head())

print("Conteo de valores nulos")
print(data.isnull().sum())

X= data[['TV', 'radio', 'newspaper']]
y= data['sales']

X_train = None
X_test = None
y_train = None
y_test = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}, {y_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

y_pred = modelo.predict(X_test_scaled)
print("\nPrimeras predicciones")
print(y_pred[:10])

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo")
print(f"Error cuadrático medio (MSE): {mse:.3f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.3f}")
print(f"Coeficiente de determinación (R²): {r2:.3f}")

residuos = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuos, alpha=0.7, edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Gráfico de residuos del modelo de regresión lineal")
plt.xlabel("Ventas predichas")
plt.ylabel("Residuos (Error de predicción)")
plt.show()