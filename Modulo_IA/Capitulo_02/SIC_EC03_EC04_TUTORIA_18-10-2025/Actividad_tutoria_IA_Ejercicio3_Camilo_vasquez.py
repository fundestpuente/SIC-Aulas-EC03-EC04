#EJERCICIO 3 
#Camilo Vasquez
import numpy as np
import sympy as sp
#Matrices
A = np.array([[2, 1], [0, 3]])
B = np.array([[1, 2], [3, 0]])
#1.- Calcular a+b y a*b
suma = A + B
print("suma de A y B:" ,suma)
producto = np.matmul(A, B)
print("producto entre A y B:" ,producto)
#2.- Calcular la inversa de A y el determinante de B
#inversa de A
inversa_A = np.linalg.inv(A)
print("inversa:" ,inversa_A)
# determinate de B
dB = np.linalg.det(B)
print("determinante de b:" ,int(dB))
#Vectores
u = np.array([1, 3])
v = np.array([-1, 2])
#1.-Producto interno U, V
producto_interno = np.dot(u, v)
print("producto interno:" ,producto_interno)
#2.- Calcular la magnitud de cada vector
magnitud_u = np.linalg.norm(u)
print("magnitud de u:" ,magnitud_u)
magnitud_v = np.linalg.norm(v)
print("magnitud de v:" ,magnitud_v)
#Derivadas
x = sp.Symbol('x')
h = sp.sin(2*x**2 + 3*x)
primera_derivada = sp.diff(h, x)
#1.-primera y segunda derivada
print("Primera derivada:" ,primera_derivada)
segunda_derivada = sp.diff(h, x, 2)
print("segunda_derivada:" ,segunda_derivada)
#2.-Evaluar en x=(0,1)
# Evaluar x=1
evaluando_1 = primera_derivada.subs(x, 1)
print("h'(1) =",evaluando_1)
# Evaluar x=0
evaluando_0 = primera_derivada.subs(x, 0)
print("h'(0) = " ,evaluando_0)
