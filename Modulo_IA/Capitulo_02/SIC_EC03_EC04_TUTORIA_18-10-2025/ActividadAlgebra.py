import numpy as np
import sympy as sp

#Julian Ruiz

#Ejercicio 1
#Matrices:
A =np.array( [[1, 2], [3, 4]])
B =np.array( [[2, 0], [1, 2]])

#Calcula A + B, A - B y el producto matricial A × B.
suma=A+B
print(suma)

resta=A-B
print(resta)

producto= np.dot(A,B)
print(producto)

#Calcula la traspuesta de A y la inversa de B (si existe).
transpuesta=A.T
print(transpuesta)

try:
    inversa= np.linalg.inv(B)
except np.linalg.LinAlgError:
    inversa = "No tiene inversa (det = 0)"
print(inversa)


#Vectores:
u = np.array([1, 2])
v = np.array([3, -1])

#Calcula el producto interno u·v.
prodInterno=np.dot(u,v)
print(prodInterno)

#Calcula la magnitud de cada vector.
uMag = np.linalg.norm(u)
print(uMag)
vMag = np.linalg.norm(v)
print(vMag)


#Derivadas:
#f(x) = x3 + 2*x2 + x
x=sp.Symbol("X")
f=x**3+2*x**2+x
print(f)

#Calcula la derivada primera y segunda de f(x).
first=sp.diff(f,x)
print(first)
second=sp.diff(first,x)
print(second)

#Evalúa ambas derivadas en x = 1. """
firstEv=first.subs(x,1)
print(firstEv)
secondEv=second.subs(x,1)
print(secondEv)