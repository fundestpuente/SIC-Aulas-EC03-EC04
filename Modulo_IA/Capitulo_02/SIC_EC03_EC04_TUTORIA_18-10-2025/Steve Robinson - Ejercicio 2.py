import numpy as np
import sympy as sp

"""
Matrices:
A = [[1, 0, 2], [2, 1, 1], [3, 2, 0]]  
B = [[0, 1, 1], [1, 2, 0], [2, 1, 3]]  

1. Calcula A × B.  
2. Calcula la traspuesta y determinante de A.
"""

A = np.array([[1, 0, 2], [2, 1, 1], [3, 2, 0]])
B = np.array([[0, 1, 1], [1, 2, 0], [2, 1, 3]])

multiplicacion = np.dot(A, B)
print("A x B =\n", multiplicacion)

transpuesta = A.T
print("La matriz transpuesta de A es:\n", transpuesta)
detA = np.linalg.det(A)
print("El determinante de A es:", detA, end="\n\n")

"""
Vectores: 
u = [2, -1, 3]  
v = [1, 4, -2]  

1. Calcula el producto interno u·v y determina si son ortogonales.  
2. Calcula el ángulo entre u y v.
"""

u = np.array([2, -1, 3])
v = np.array([1, 4, -2])

producto_punto = np.dot(u, v)
print("u·v =", producto_punto)
if producto_punto == 0:
    print("Los vectores A y B son ortogonales")
else:
    print("Los vectores A y B no son ortogonales")

angulo_vectores = np.arccos(producto_punto / (np.linalg.norm(u) * np.linalg.norm(v)))
print("El ángulo en radianes entre los vectores A y B es:", angulo_vectores, end="\n\n")

"""
Derivadas:  
g(x, y) = x**2 * y + sin(x*y)  

1. Calcula las derivadas parciales ∂g/∂x y ∂g/∂y.  
2. Evalúa en el punto (x=1, y=π/2).
"""

x = sp.Symbol("x")
y = sp.Symbol("y")
func_g = x**2 * y + sp.sin(x*y)

derivada_parcial_x = sp.diff(func_g, x)
print("∂g/∂x =", derivada_parcial_x)

derivada_parcial_y = sp.diff(func_g, y)
print("∂g/∂y =", derivada_parcial_y)

evaluacion = func_g.evalf(subs={x: 1, y: np.pi/2})
print("La función g evaluada en el punto (x=1, y=π/2) vale:", evaluacion)