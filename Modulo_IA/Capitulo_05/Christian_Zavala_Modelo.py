import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos
url= "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df = pd.read_csv(url)

print("Primeras filas del conjunto de datos:")
print(df.head())

print("\nInformación General del conjunto de datos:")
print(df.info())

#Limpiar datos (si es necesario)

print("\nValores nulos en cada columna:")
print(df.isnull().sum())
# No hay valores nulos en este conjunto de datos

# Seleccionar características y variable objetivo
x= df.drop(['sales', 'Unnamed: 0'], axis=1) # Características
y= df['sales'] # Variable objetivo

# Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTamaño del conjunto de entrenamiento:", x_train.shape)
print("Tamaño del conjunto de prueba:", x_test.shape)
# Escalar las características
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x_train_scaled, y_train)
# Hacer predicciones
y_pred = model.predict(x_test_scaled)
# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
mape = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nEvaluación del modelo:")
print("Error Cuadrático Medio (MSE):", mse)
print("Raíz del Error Cuadrático Medio (RMSE):", mape)
print("Coeficiente de Determinación (R²):", r2)

# Visualizar los resultados
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales de Ventas")
plt.ylabel("Valores Predichos de Ventas")
plt.title("Valores Reales vs Predichos de Ventas")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.show()