import numpy as np
from sympy import symbols, diff

# -- Ejercicios con Matrices --

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

#1. Cálculos básicos de matrices
suma_AB = A + B
resta_AB = A - B
producto_AB = np.dot(A, B)

#2. Transpuesta e Inversa
transpuesta_A = A.T

#Verificamos si la inversa de B existe (si el determinante no es cero)
if np.linalg.det(B) != 0:
    inversa_B = np.linalg.inv(B)
else:
    inversa_B = "La matriz B no tiene inversa."

print("--- Operaciones con Matrices ---")
print("Matriz A:\n", A)
print("\nMatriz B:\n", B)
print("\nSuma A + B:\n", suma_AB)
print("\nResta A - B:\n", resta_AB)
print("\nProducto Matricial A x B:\n", producto_AB)
print("\nTranspuesta de A:\n", transpuesta_A)
print("\nInversa de B:\n", inversa_B)


# -- Ejercicios con Vectores --

u = np.array([1, 2])
v = np.array([3, -1])

#1. Producto interno
producto_interno_uv = np.dot(u, v)

#2. Norma
magnitud_u = np.linalg.norm(u)
magnitud_v = np.linalg.norm(v)

print("\n\n--- Operaciones con Vectores ---")
print("Vector u:", u)
print("Vector v:", v)
print("\nProducto Interno u · v:", producto_interno_uv)
print("\nMagnitud de u:", magnitud_u)
print("Magnitud de v:", magnitud_v)


# --Ejercicios con Derivadas--

x = symbols('x')
f_x = x**3 + 2*x**2 + x

#1. Cálculo de derivadas
primera_derivada = diff(f_x, x)
segunda_derivada = diff(primera_derivada, x)

#2. Evaluación de las derivadas en x = 1
valor_primera = primera_derivada.subs(x, 1)
valor_segunda = segunda_derivada.subs(x, 1)

print("\n\n--- Cálculo de Derivadas ---")
print(f"Función f(x) = {f_x}")
print("\nPrimera Derivada f'(x):", primera_derivada)
print("Segunda Derivada f''(x):", segunda_derivada)
print("\nPrimera Derivada evaluada en x = 1:", valor_primera)
print("Segunda Derivada evaluada en x = 1:", valor_segunda)