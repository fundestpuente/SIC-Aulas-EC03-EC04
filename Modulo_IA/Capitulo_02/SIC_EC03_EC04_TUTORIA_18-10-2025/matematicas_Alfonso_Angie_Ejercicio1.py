import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# EJERCICIO 1

'''Matrices:

A = [[1, 2], [3, 4]]  
B = [[2, 0], [1, 2]]  

'''
print("Operaciones con matrices A y B")
print("A = [[1, 2], [3, 4]]")
print("B = [[2, 0], [1, 2]]")
print("------------------------------------------")

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

#1. Calcula A + B, A - B y el producto matricial A × B.  
sumaAB = A + B
diferenciaAB = A - B
productoAB = np.dot(A, B)

print("A + B =\n", sumaAB)
print("A - B =\n", diferenciaAB)
print("A × B =\n", productoAB)

#2. Calcula la traspuesta de A y la inversa de B (si existe).

transpuestaA = np.transpose(A)
if np.linalg.det(B) != 0:
    inversaB = np.linalg.inv(B)
else:
    inversaB = "No existe" 

print("Traspuesta de A =\n", transpuestaA)
print("Inversa de B =\n", inversaB)




'''Vectores:

u = [1, 2]  
v = [3, -1]  

'''  
u = np.array([1, 2])
v = np.array([3, -1])

print("\nOperaciones con vectores u y v")
print("u =", u)
print("v =", v)
print("------------------------------------------")

#1. Calcula el producto interno u·v.  

productoIn = np.dot(u, v)
print("Producto interno u·v =", productoIn)


#2. Calcula la magnitud de cada vector.  

magnitudU = np.linalg.norm(u)
magnitudV = np.linalg.norm(v)
print("Magnitud de u =", magnitudU)
print("Magnitud de v =", magnitudV)


'''Derivadas: '''  
#f(x) = x**3 + 2*x**2 + x  
print("\nDerivadas de f(x) = x**3 + 2*x**2 + x")
print("------------------------------------------")

#1. Calcula la derivada primera y segunda de f(x).  

x = sp.symbols('x')
f = x**3 + 2*x**2 + x
deriv1= sp.diff(f, x)
deriv2 = sp.diff(deriv1, x)
print("Derivada primera f'(x) =", deriv2)
print("Derivada segunda f''(x) =", deriv2)


#2. Evalúa ambas derivadas en x = 1.

deriv1Eval = deriv1.subs(x, 1)
deriv2Eval = deriv2.subs(x, 1)
print("f'(1) =", deriv1Eval)
print("f''(1) =", deriv2Eval)




