#Daniel Heredia Ejercicio 4

import numpy as np
import sympy as sp

A = np.array([[1, 2, 0],
              [0, 1, 1],
              [3, 0, 2]])

B = np.array([[2, 1, 1],
              [1, 0, 3],
              [0, 2, 1]])

Producto_AB = np.dot(A, B)
B_transpuesta = B.T

print("El producto de A y B es =\n", Producto_AB )
print("La transpuesta de B es =\n", B_transpuesta)

#EJERCICIO 2:

determinante_A = np.linalg.det(A) 
print("El determinante de A es=", determinante_A)

if  determinante_A != 0:
    invA = np.linalg.inv(A)
    print("La inversa de A es=\n", invA)
else:
    print("La matriz A no tiene inversa (el determinante es cero)")
    
#Ejercicio 3

u = np.array([1, -2, 3])
v = np.array([0, 1, 4])

producto_p = np.dot(u, v)
print(" El producto interno u·v =", producto_p )

mag_u = np.linalg.norm(u)
mag_v = np.linalg.norm(v)
cos_theta =  producto_p/ (mag_u * mag_v)
print("Modulo de U =", mag_u)
print("Modulo de V =", mag_v)
print("Angulo entre U y V =", cos_theta)

#Ejercicio 4

x = sp.Symbol('x')
f = x**4 - 3*x**3 + 2*x - 5

df = sp.diff(f, x)
ddf = sp.diff(df, x)
print("f(x) =", f)
print("f'(x) =", df)
print("f''(x) =", ddf)

df_2 = df.subs(x, 2)
ddf_2 = ddf.subs(x, 2)

print("f'(2) =", df_2)
print("f''(2) =", ddf_2)

if ddf_2 > 0:
    print(" La función es cóncava hacia arriba en x=2")
elif ddf_2 < 0:
    print("La función es cóncava hacia abajo en x=2")

