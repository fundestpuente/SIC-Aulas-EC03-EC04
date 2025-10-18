import numpy as np
import sympy as sp
# matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[2, 0],
              [1, 2]])

SumaAB = A + B
RestaAB = A - B
MultiplicarAB = np.matmul(A, B)
TranspuestaA = A.T

if np.linalg.det(B) != 0:
    B_inv = np.linalg.inv(B)
else:
    B_inv = None

# vectores
u = np.array([1, 2])
v = np.array([3, -1])

internoUV = np.dot(u, v)
magnitud_u = np.linalg.norm(u)
magnitud_v = np.linalg.norm(v)

# derivadas
x = sp.symbols('x')
f = x**3 + 2*x**2 + x

f_derivada = sp.diff(f, x)
f_dobleDerivada = sp.diff(f, x, 2)

f_derivada_1 = sp.N(f_derivada.subs(x, 1))
f_dobleDerivada_1 = sp.N(f_dobleDerivada.subs(x, 1))

print("MATRICES")
print("A =", A)
print("B =", B)
print("A + B =", SumaAB)
print("A - B =", RestaAB)
print("A × B =", MultiplicarAB)
print("Traspuesta de A (A^T) =", TranspuestaA)
print("Inversa de B =", B_inv if B_inv is not None else "No existe (det=0)")

print("\nVECTORES ")
print("u =", u)
print("v =", v)
print("u · v =", internoUV)
print("|u| =", magnitud_u)
print("|v| =", magnitud_v)

print("\nDERIVADAS ")
print("f(x) =", f)
print("f'(x) =", f_derivada)
print("f''(x) =", f_dobleDerivada)
print("f'(1) =", f_derivada_1)
print("f''(1) =", f_dobleDerivada_1)