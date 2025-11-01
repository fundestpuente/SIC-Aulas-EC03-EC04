import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

url ="https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df= pd.read_csv(url, sep =",")

print(f"Cantidad de valores nulos:\n {df.isnull().sum()}")
print("Columnas",df.columns.tolist())
#primeras 5 
print("Primeros 5 datos\n",df.head())
x =df.drop(['Unnamed: 0', 'sales'], axis=1) # variable X
y = df['sales'] #variable Y
# 20% de los datos van a test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
#normalizacion
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#modelo, se usa las variables de entrenamiento (X y Y)
model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
print(f"Primeras predicciones: \n {y_pred[:10]}")

mse = mean_squared_error(y_test, y_pred) #error cuadratico medio
rmse = np.sqrt(mse) #Raiz de error cuadratico medio
r2 = r2_score(y_test, y_pred) # coeficiente de determinacion r^2

print(f"Error cuadratico medio: {mse}")
print(f"Raiz de error cuadratico medio: {rmse}")
print(f" Coeficiente de determinacion R^2: {r2}")

residuos = y_test - y_pred
plt.figure(figsize =(8,5))
plt.scatter(y_pred, residuos, alpha = 0.7,edgecolors = 'k' )
plt.axhline(y = 0 ,color = 'red', linestyle ='--', linewidth=2)
plt.title("Grafico de residuos del modelo de regresión lineal")
plt.xlabel("Valores predichos al ingreso de las ventas")
plt.ylabel("Residuos")
plt.show()
#Interpretacion, segun la grafica, el modelo puede predecir las ventas con bastante precision,
#  y ya que los errores son distribuidos aleatoriamente, es posible que la relacion
#  de medio de publicidad y ventas, es que, mientras mas inversion en publicidad, mejor ventas.

