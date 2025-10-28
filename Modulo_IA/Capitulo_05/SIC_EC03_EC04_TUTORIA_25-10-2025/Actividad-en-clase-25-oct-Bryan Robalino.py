import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt

# lograr predecir las ventas futuras a partir de las ventas de acda mes hacer un analisis de lo que pueda con eso

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

df = pd.read_csv(url)

print("Vista general del dataset:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

# ============================================
# Preparar los datos
# ============================================
# Variables independientes (X)
X = df[["TV", "radio", "newspaper"]]

# Variable dependiente (Y)
y = df["sales"]

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# ============================================
# Escalar las variables
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# Crear y entrenar el modelo
# ============================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ============================================
# Realizar predicciones
# ============================================
y_pred = model.predict(X_test_scaled)
print("\nPrimeras predicciones:")
print(y_pred[:10])

# ============================================
# Evaluar el modelo
# ============================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo:")
print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")

# ============================================
# Coeficientes del modelo
# ============================================
coeficientes = pd.DataFrame({
    "Variable": ["TV", "Radio", "Newspaper"],
    "Coeficiente": model.coef_
})
print("\nCoeficientes del modelo:")
print(coeficientes)
print(f"Intercepto (b): {model.intercept_:.2f}")

# ============================================
# Gráficos para análisis
# ============================================

# Gráfico 1: valores reales vs predichos
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='k')
plt.xlabel("Ventas reales")
plt.ylabel("Ventas predichas")
plt.title("Comparación entre ventas reales y predichas")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # línea ideal

# Gráfico 2: residuos
residuos = y_test - y_pred
plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuos, alpha=0.7, edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title("Gráfico de residuos del modelo")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.show()